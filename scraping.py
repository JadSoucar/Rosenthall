#STANDARD LIBRARY
import os
import re
import datetime
from multiprocessing import Manager
from multiprocessing import Pool
import math
import pickle

#IMPORTED LIBRARIES
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import pymysql
pd.set_option('mode.chained_assignment', None)
from tqdm import tqdm
from pandarallel import pandarallel


global ipv4
ipv4 = "___"


def get_recording_info(file):
	import re
	import enchant

	try:
		#Imputed Information 
		year = re.search('\d\d\d\d',file).group()
		month = re.search('(?<=-)\d\d(?=-)',file).group()
		day = re.search('(?<=-)\d\d(?=_)',file).group()
		tape_num_obj = re.search("(?<=US_).*?(?=_)",file)
		tape_num = tape_num_obj.group()
		v_letter = file[tape_num_obj.span()[-1]+1]
		v_number = re.search("(?<=\w)\d*?(?=_)",file[tape_num_obj.span()[-1]+1:]).group()
		file_ext = re.search("(?<=\.).*",file).group()
		
		#Content Information 
		with open('GetFiles/all_files/'+file,'r', errors='backslashreplace') as f:
			content = f.read()
		dur = re.search("(?<=DUR\|).*(?=\n)",content).group()
		cc = re.findall('(?<=\|CC1\|).*(?=\n)',content)
		d = enchant.Dict("en_US")
		total_word_count = 0
		total_correct = 0
		for line in cc:
			split_line = line.split(' ')
			candidate_words = []
			real_words = []
			for word in split_line:
				if word != '':
					clean_word = word.lower().strip(' ')
					candidate_words.append(clean_word)
					if d.check(clean_word):
						real_words.append(clean_word)  
			total_word_count += len(candidate_words)
			total_correct += len(real_words) 


		return (f'{year}-{month}-{day}', tape_num, v_letter, v_number, re.sub('\..*?$','',file), file_ext, dur, total_word_count, total_correct)
	except Exception as e:
		print(e)
		file_ext = re.search("(?<=\.).*",file).group()
		return ('0-0-0', '0', '0', 0, re.sub('\..*?$','',file), file_ext, 0, 0, 0)


def get_bound(line):
    candidates = []
    candidates.append([re.search('SEG_00\|Type=Commercial',line),'txt_seg_comm'])
    candidates.append([re.search('SEG_00\|Type=Story start',line),'txt_seg_story_start'])
    candidates.append([re.search('SEG_00\|Type=Story end',line),'txt_seg_story_end'])
    candidates.append([re.search('captio',line.lower()),'txt_caption'])
    for candidate,candidate_type in candidates:
        if candidate != None:
            start = re.search('^.*?(?=\|)',line).group()
            end = re.search('(?<=\|).*?(?=\|)',line).group()
            start = re.sub('-','',start)
            end = re.sub('-','',end)
            return [start, end, candidate_type] 

    return None

def get_context(line):
    match = re.search('(?<=\|CC1\|).*(?=\n)',line)
    if match != None:
        clean = re.sub('[^\d^\w^\s]','',match.group())
        clean = clean.strip(' ')
        clean = clean.replace('\xa0','')
        clean = clean.replace('â','')
        return clean
    
    return None

def get_boundary_info(tape):
    tape_id = tape[0]
    tape_num = tape[2]
    tape_file = tape[5]
    with open('GetFiles/all_files/'+tape_file+'.txt3','r', errors='backslashreplace') as f:
        content = f.readlines()
        
    candidate_bounds = []
    for ix,line in enumerate(content):
        line_bound = get_bound(line)
        
        if line_bound != None:
            bound_context = []
            for i in range(-3,4,1):
                if (ix+i >= len(content)) or (ix+i < 0):
                    bound_context.append(None)
                else:
                    bound_context.append(get_context(content[ix+i]))
                    
            line_bound.append(bound_context)
            candidate_bounds.append(line_bound)
    
    candidate_bounds = pd.Series(candidate_bounds, dtype = object)
    bounds = candidate_bounds[candidate_bounds.apply(lambda x: x != None)]
    bounds.apply(lambda x: x.append(tape_id))
    return bounds.tolist()


def get_total_seconds(x):
	hrs = x[8:10]
	mins = x[10:12]
	secs = x[12:14]
	milisecs = x[15:]
	return int(hrs)*60*60 + int(mins)*60 + int(secs) + float('0.'+milisecs)

def assign_clusters_to_data(data,day_numbers, v_numbers, min_count,linkage_thresh):
	cluster_sizes = {}
	clustering_codes = {}
	for v_num in v_numbers:
		for wk_day in tqdm(day_numbers):
			#filter on day and v number
			day_data = data[data['V_NUMBER'] == v_num]
			day_data = day_data[day_data['DATE_PULLED'].dt.weekday == wk_day]
			#Ordinal Transform on Date
			day_data['DATE_PULLED'] = day_data['DATE_PULLED'].map(datetime.datetime.toordinal)
			if len(day_data)>2:
				#Cluster
				X = day_data[['DATE_PULLED','BOUNDARY_START_TIME']].copy(deep = True)
				clustering = AgglomerativeClustering(linkage='single',n_clusters=None,distance_threshold=linkage_thresh,metric='euclidean').fit(X.values)

				#Get Clusters sizes
				for i in set(clustering.labels_):
					cluster_sizes[f'V{v_num}_{wk_day}_{i}'] = list(clustering.labels_).count(i)

				#Get Cluster Codes for Corresponding Boundary Ids
				for boundary_id,cluster_id in zip(day_data['BOUNDARY_ID'].values.tolist(),clustering.labels_):
					clustering_codes[boundary_id] = f'V{v_num}_{wk_day}_{cluster_id}'
			else:
				#Get Cluster Codes for Corresponding Boundary Ids
				for boundary_id in day_data['BOUNDARY_ID'].values.tolist():
					clustering_codes[boundary_id] = f'V{v_num}_{wk_day}_NA'
					
	return clustering_codes, cluster_sizes


def get_scores(ixs,ns):
	from clustering import ContextScore
	cs = ContextScore()

	scores = []
	for ix in ixs:
		boundary, year, data = ns.data_with_clusters.loc[ix], 1, ns.data_with_clusters
		score = cs.get_context_score(boundary,year,data)
		scores.append(score)

	return scores

def chunked(iterable, n):
    chunksize = int(math.ceil(len(iterable) / n))
    return [iterable[i * chunksize:i * chunksize + chunksize] for i in range(n)]

def chunks(lst, n):
	chunks = []
	for i in range(0, len(lst), n):
		chunks.append(lst[i:i + n])
	return chunks

def multiprocessed_get_scores(n_cpu,data_with_clusters):
	n_cpu = 8
	#we are chunking the df so the shared memory doesnt get overloaded with a giant dataframe, which slows down the process
	dfs = chunked(data_with_clusters,round(len(data_with_clusters)/1000))
	dfs_with_scores = []
	for df in tqdm(dfs):
		mgr = Manager()
		ns = mgr.Namespace()
		ns.data_with_clusters = df
		pool = Pool(processes=n_cpu)
		ix_batches = chunked(ns.data_with_clusters.index,n_cpu)
		result = pool.starmap(get_scores,zip(ix_batches,[ns for _ in range(n_cpu)]))
		pool.close()
		scores = []; [scores.extend(x) for x in result] 
		temp = pd.DataFrame.from_dict(dict(scores), orient='index')
		temp.columns = ['CONTEXT_SCORE']
		data_with_scores = ns.data_with_clusters.join(temp, on='BOUNDARY_ID')
		dfs_with_scores.append(data_with_scores)


	data_with_scores = pd.concat(dfs_with_scores)
	return data_with_scores

def assign_cluster_class(cluster_code,context_score,min_count,score_thresh,cluster_sizes):
    '''
	The Three Boundary Clases Are
	1. incluster-True: In Cluster and Context is similar to other points in cluster 
	2. incluster-False: In Cluster and Context is NOT similar to other points in cluster
	3. notincuster: Not in a cluster
	'''

    if 'NA' in cluster_code:
    	#this will only happen if there is less then 2 boundaries for a full day on a V number
        return 'not-in-cluster'
    elif cluster_sizes.loc[cluster_code]['CLUSTER_SIZE'] < min_count:
        return 'not-in-cluster'
    elif context_score < score_thresh:
        return 'in-cluster-False'
    else:
        return 'in-cluster-True'


def scrape_tape():
	pandarallel.initialize(nb_workers=16, progress_bar = True)
	with open('GetFiles/txt_file.pkl','rb') as f:
		files =  pickle.load(f)

	with open('BackUp/file_info.pkl','wb') as f:
		pickle.dump([],f)

	batches = chunks(files, 1000)
	for batch in tqdm(batches):
		with open('BackUp/file_info.pkl','rb') as f:
			file_info =  pickle.load(f)

		info = pd.Series(batch).parallel_apply(get_recording_info)
		file_info.extend(info.tolist())

		with open('BackUp/file_info.pkl','wb') as f:
			pickle.dump(file_info,f)

	return True


def input_tape():
	#Connect to Server
	conn = pymysql.connect(host=ipv4,port=3307,user='root',password='rosemax',db='boundary_detection')
	cur = conn.cursor()
	with open('BackUp/file_info.pkl','rb') as f:
		file_info =  pickle.load(f)

	#Insert from pickle file
	bad_files = []
	query = "INSERT INTO TAPE (DATE_PULLED, TAPE_NUMBER, V_LETTER, V_NUMBER, FILE_NAME, FILE_EXT, DURATION, CC_COUNT, CC_CORRECT_COUNT) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
	for file in tqdm(file_info):
		try:
			cur.execute(query,file)
		except Exception:
			bad_files.append(file)
	conn.commit()
	print('Commited')
	with open('BackUp/badfiles_tape.pkl','wb') as f:
		pickle.dump(bad_files,f)


	return True


def scrape_boundary():
	pandarallel.initialize(nb_workers=16, progress_bar = True)
	conn = pymysql.connect(host=ipv4,port=3307,user='root',password='rosemax',db='boundary_detection')
	cur = conn.cursor()
	query = 'SELECT * FROM TAPE'
	cur.execute(query)
	tapes = cur.fetchall()

	batches = chunks(tapes, 1000)

	ix = 1
	
	with open(f'BackUp/boundary_info{ix}.pkl','wb') as f:
		pickle.dump([],f)
	
	
	for batch in tqdm(batches):
		with open(f'BackUp/boundary_info{ix}.pkl','rb') as f:
			boundaries =  pickle.load(f)

		bounds = pd.Series(batch).parallel_apply(get_boundary_info)
		boundaries.extend(bounds.tolist())

		with open(f'BackUp/boundary_info{ix}.pkl','wb') as f:
			pickle.dump(boundaries,f)

		del boundaries
		del bounds

		if os.path.getsize(f'BackUp/boundary_info{ix}.pkl')/1000 >= 500_000:
			ix = ix + 1
			with open(f'BackUp/boundary_info{ix}.pkl','wb') as f:
				pickle.dump([],f)


	return True


def input_boundary():
	#https://dev.to/qviper/connecting-mysql-server-in-windows-machine-from-wsl-4pf1#:~:text=Connecting%20MySQL%20Server%20in%20Windows%20Machine%20from%20WSL,Call%20from%20WSL%20...%204%20From%20WSL%20
	#how to run pymysql on linux --> building this docker would be huge I need maria
	conn = pymysql.connect(host=ipv4,port=3307,user='root',password='rosemax',db='boundary_detection')
	cur = conn.cursor()

	for ix in range(1,4):

		with open(f'BackUp/boundary_info{ix}.pkl','rb') as f:
			boundary_info =  pickle.load(f)

		query = "INSERT INTO BOUNDARY (BOUNDARY_START_TIME, BOUNDARY_END_TIME, BOUNDARY_SOURCE, CONTEXT, TAPE_ID) VALUES (%s, %s, %s, %s, %s)"	
		bad_files = []
		for boundary_batch in tqdm(boundary_info):
			for boundary in boundary_batch:
				try:
					boundary[3] = ', '.join(list(map(lambda x: 'None' if x==None else x,boundary[3])))	
					cur.execute(query,tuple(boundary))
				except Exception as e:
					print(e)
					bad_files.append(boundary)

		
		conn.commit()
		with open(f'BackUp/badfiles_boundary{ix}.pkl','wb') as f:
			pickle.dump(bad_files,f)


	return True


def search(string):
    string = string.lower()
    p1 = re.search("(closed )?caption(s|ing) sponsored (by )?[.^\S]*",string)
    p2 = re.search("(closed )?caption(s|ing) paid for (by )?[.^\S]*",string)
    p3 = re.search("(closed )?caption(s|ing) performed (by )?[.^\S]*",string)
    p4 = re.search("(closed )?caption(s|ing) provided (by )?[.^\S]*",string)
    p5 = re.search("captions prohibited without [.^\S]*",string)
    p6 = re.search("(closed )?caption(s|ing) made (possible )?(by )?[.^\S]*",string)
    p7 = re.search("(closed )?caption(s|ing|ed) by [.^\S]*",string)
    p8 = re.search("caption(ing|s)? services",string)
   
    matches = []
    for ix,p in enumerate([p1,p2,p3,p4,p5,p6,p7,p8]):
        if p!=None:
            return ix+1
    return 0


def scrape_cluster(n_cpu = 8,min_count = 20,linkage_thresh = 100,score_thresh = .6):
	'''
	n_cpu -> number of cpus to use for multiprocessing
	min_count -> min number of boundaries in a cluster to be considered a valid cluster
	linkage_thresh -> the maximum distance points in a cluster can be from its nearests neighbor
	score_thres -> the minimum context_score for a point to be considered a true member of a cluster 
				   (view def assign_cluster_class for more detail)
	'''

	#GET DATA FROM SQL
	#Get MariaDB Connection
	conn = pymysql.connect(host=ipv4,port=3307,user='root',password='rosemax',db='boundary_detection')
	#focus in on boundaries found using "captio" method and that have a V Number with 2 digits
	query = f"select * from tape inner join boundary on tape.tape_id = boundary.tape_id where tape.v_letter = 'V' and boundary.boundary_source = 'txt_caption'"
	data = pd.read_sql(query, conn)

	#CLEAN DATA FROM SQL
	#Get time in seconds
	data['BOUNDARY_START_TIME'] = data['BOUNDARY_START_TIME'].apply(get_total_seconds)
	data['BOUNDARY_END_TIME'] = data['BOUNDARY_END_TIME'].apply(get_total_seconds)
	#turn date to datetime
	data['DATE_PULLED'] = pd.to_datetime(data['DATE_PULLED'])
	#Only focus on cleaner data from 1996-2006
	data = data[data['DATE_PULLED'].dt.year.isin(range(1996,2006))]
	v_numbers = list(set(data['V_NUMBER']))
	day_numbers = list(range(0,8))

	#assign clusters to boundaries
	clustering_codes, cluster_sizes  = assign_clusters_to_data(data,day_numbers = day_numbers, v_numbers = v_numbers, min_count = min_count,linkage_thresh = linkage_thresh) 
	temp = pd.DataFrame.from_dict(clustering_codes,orient='index')
	temp.columns = ['CLUSTER_ID']
	data_with_clusters = data.join(temp, on='BOUNDARY_ID')
	cluster_sizes = pd.DataFrame.from_dict(cluster_sizes, columns = ['CLUSTER_SIZE'], orient='index')

	data_with_clusters.to_pickle('data_with_clusters.pkl')

	#get template id
	stringify = lambda context_str: ' '.join([word.strip(' ') for word in list(reversed(context_str.split(',')))])
	all_contexts = data_with_clusters['CONTEXT'].apply(stringify)
	matches = all_contexts.apply(search)
	data_with_clusters['TEMPLATE_ID'] = matches

	#assign context scores to boundaries
	#data_with_scores = multiprocessed_get_scores(n_cpu,data_with_clusters)

	#assing cluster classes to boundaries
	#data_with_scores['CLUSTER_CLASS'] = data_with_scores[['CLUSTER_ID','CONTEXT_SCORE']].apply(lambda x: assign_cluster_class(cluster_code = x[0], context_score = x[1],min_count=min_count,score_thresh=score_thresh,cluster_sizes=cluster_sizes), axis=1)
	
	#Save to BackUp
	data_with_clusters.to_pickle('BackUp/data_with_scores.pkl')

	return True


def input_cluster():
	conn = pymysql.connect(host=ipv4,port=3307,user='root',password='rosemax',db='boundary_detection')
	cur = conn.cursor()
	data = pd.read_pickle('BackUp/data_with_scores.pkl')

	query = "INSERT INTO CONFIDENCE (BOUNDARY_ID, CLUSTER_ID, TEMPLATE_ID) VALUES (%s, %s, %s)"	
	bad_files = []
	for ix in tqdm(data.index):
		boundary = data.loc[ix]
		cur.execute(query,(boundary['BOUNDARY_ID'],boundary['CLUSTER_ID'],boundary['TEMPLATE_ID']))

	conn.commit()

	return True




if __name__=='__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--scrape_tape', action = 'store_true')
	parser.add_argument('--scrape_boundary', action = 'store_true')
	parser.add_argument('--scrape_cluster', action = 'store_true')
	parser.add_argument('--input_tape', action = 'store_true')
	parser.add_argument('--input_boundary', action = 'store_true')
	parser.add_argument('--input_cluster', action = 'store_true')
	args = parser.parse_args()

	if args.scrape_tape:
		scrape_tape()

	if args.input_tape:
		input_tape()
		
	if args.scrape_boundary:
		scrape_boundary()

	if args.input_boundary:
		input_boundary()

	if args.scrape_cluster:
		scrape_cluster()

	if args.input_cluster:
		input_cluster()


#from a pipline perspective what needs to be done is getting functionality to get all filles
#so that if i wanted to I could populate every table from scratch 
#so that once they have the sql queries they need to run to get top prospects they will have everything they need
