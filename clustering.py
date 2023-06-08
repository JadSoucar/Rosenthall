from difflib import SequenceMatcher
import datetime
import pandas as pd
		
class ContextScore:
	@staticmethod
	def compare_context_wharped(c1,c2,n_lines):
		max_sim = []
		for i in range(n_lines):
			if c1[i] != None:
				j_sims = []
				for j in range(n_lines):
					if c2[j] != None:
						j_sims.append(SequenceMatcher(None, c1[i], c2[j]).ratio())
				max_sim.append(max(j_sims))  
		
		return sum(max_sim)/len(max_sim)

	def get_context_score(self,boundary, years, data):
		cluster_data = data[data['CLUSTER_ID']==boundary['CLUSTER_ID']]
		boundary_id = boundary['BOUNDARY_ID']
		if len(cluster_data) == 1:
			return (boundary_id,1.0)
		
		radius = (years*365)/2
		center = boundary['DATE_PULLED']
		right_end = center + datetime.timedelta(days=radius)
		left_end  = center - datetime.timedelta(days=radius)
		ball = cluster_data[(cluster_data['DATE_PULLED']>left_end).to_numpy() & (cluster_data['DATE_PULLED']<right_end).to_numpy()]
		boundary_context = [word.strip(' ') for word in boundary['CONTEXT'].split(',')]
		context_scores = []
		for bound in ball.itertuples():
			bound_context = [word.strip(' ') for word in bound[16].split(',')]
			bound_id = bound[11]
			if bound_id != boundary_id:
				context_score = self.compare_context_wharped(boundary_context,bound_context,7)
				context_scores.append(context_score)
		
		return (boundary_id ,sum(context_scores)/len(context_scores))