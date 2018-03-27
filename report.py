'''
the Reporter class is use to help log, statistics, debug, measure process time

execute and initialize a instance named reporter when initial import by parser_main.py

'''

#for measuring process time 
import time

import sys
from sklearn.metrics import precision_recall_fscore_support

#blank buffer for reuse the output line in Reporter.log
N_BLANK_BUFFER = 10

class Reporter():
	def __init__(self):
		#for measuring process time
		self.time = 0.0
		
		self.parse_eval_data_dict = {'merge': {}, 'sense': {}, 'center': {}, 'join': {}}
		for key in self.parse_eval_data_dict:
			self.parse_eval_data_dict[key] = {'tp':0, 'gold_n':0, 'pred_n':0}
		self.gold_edu_tag_list = []
		self.pred_edu_tag_list = []
		#memo: 
		#some paragraphs from xml is empty
		#some relation spans from xml is not post order
	
	#output the object's attributes, may be used for the Corpus object
	#the object_attr is the attribute name string(ex: 'relations') that itself is also an object(like the Relation objects in Corpus)
	def show_attributes(self, obj, object_attr=None, simplify_attr=None):
		for attr, value in obj.__dict__.iteritems():
			#for the relation object
			if attr == object_attr:
				print attr,':'
				#output the object attribute's attribute, and deal with the condition that there is a list of the objects(like the Relation objects in Corpus )
				if type(value) == type([]):
					for v_obj in value:
						self.show_attributes(v_obj)
				else:
					self.show_attributes(value)
			elif attr == simplify_attr:
				print attr,':', 
				if type(value) == type([]):
					for v in value:
						print type(v), ' '
				else:
					print type(value)
			else:
				if type(value) == type(u''):
					value = value.encode('utf-8')
				print attr,':', value
		
	
	#log what the caller function is doing, format: "<message> [from <src>]..."
	#if repeat=True, reusing a line when called repeatedly
	def log(self, message, src=None, repeat=False):		
		if src != None:
			message = message + ' from ' + src
		
		message += '...'
		
		#if repeat=True, reusing a line when called repeatedly
		if repeat:
			#blank buffer for reuse the output line
			message += ' '*N_BLANK_BUFFER
			#to reuse the output line
			message  = '\r' + message 
		else:
			message += '\n'

		#if repeat=False, a new line is already appended
		print message,
		
	
	def show_eval_result(self):
		
		edu_result = precision_recall_fscore_support(self.gold_edu_tag_list, self.pred_edu_tag_list, average='binary')
		print edu_result
		for key, dict in self.parse_eval_data_dict.iteritems():
			print key,': '
			for k, v in dict.iteritems():
				print k, ': ', v
			print 'f1: ', 2*float( dict['tp'] )/( dict['gold_n'] + dict['pred_n'] )
		print edu_result

			
	
	def collect_eval_data(self, gold_instance, pred_instance):
		#self.show_attributes(gold_instance, object_attr='i_relations',simplify_attr='segments')
		#self.show_attributes(pred_instance, object_attr='i_relations',simplify_attr='segments')
		
		self.get_edu_eval_data(gold_instance, pred_instance)
		self.get_parse_eval_data(gold_instance, pred_instance)

	def get_edu_eval_data(self, gold_instance, pred_instance):
		
		gold_edu_boundary_list = self.get_edu_boundary_list_from_instance(gold_instance)
		pred_edu_boundary_list = self.get_edu_boundary_list_from_instance(pred_instance)
		
		gold_edu_tag_list = []
		pred_edu_tag_list = []
		punc_list = []
		
		for segment in gold_instance.segments:
			punc_list.append(segment[-1].encode('utf-8'))
		
		seg_boundary_list = [ span[-1] for span in gold_instance.segments_span ]
		for boundary in seg_boundary_list:
			if boundary in gold_edu_boundary_list:
				gold_edu_tag_list.append(1)
			else:
				gold_edu_tag_list.append(0)
			if boundary in pred_edu_boundary_list:
				pred_edu_tag_list.append(1)
			else:
				pred_edu_tag_list.append(0)

		
		for tag, punc in zip(gold_edu_tag_list,punc_list):
			print tag, punc,
		print ''
		for tag, punc in zip(pred_edu_tag_list,punc_list):
			print tag, punc,	
		print '\n\n'
		#print gold_edu_tag_list
		#print pred_edu_tag_list
		
		self.gold_edu_tag_list.extend(gold_edu_tag_list)
		self.pred_edu_tag_list.extend(pred_edu_tag_list)
		
	def output_cky_table(self, cky_table):
		for cky_range in range(1,n_segments):
			for start_idx in range(0, n_segments-cky_range):				
				print 'start_idx:', start_idx, 'cky_range: ', cky_range
				for cky_candidate in cky_table[start_idx][cky_range].cky_candidate_list:
					for info in cky_candidate.cky_span_infos:
						print 'info_start_idx:', info.start_idx, 'info_cky_range:', info.cky_range, 'info_candidate_idx:', info.candidate_idx
					print 'representation:', type(cky_candidate.representation)
					print ''
	
	def get_edu_boundary_list_from_instance(self, instance):
		boundary_list = []
		boundary_idx_set = set()
		for i_relation in instance.i_relations:
			#span example: [ [0, 21], [22, 76] ]
			for span_unit in i_relation.span:
				boundary_idx_set.add(span_unit[-1])
		
		#if only one edu, no i_relation
		if len(instance.i_relations) == 0:
			boundary_idx_set.add(instance.segments_span[-1][-1])
		
		#sorted from left to right
		boundary_list.extend( sorted(boundary_idx_set) )
		
		return boundary_list
		
	
	def get_parse_eval_data(self, gold_instance, pred_instance):
	
		for key in self.parse_eval_data_dict:
			# use max() in case of only one edu
			self.parse_eval_data_dict[key]['gold_n'] += max(0,len(gold_instance.i_relations)-1)
			self.parse_eval_data_dict[key]['pred_n'] += max(0,len(pred_instance.i_relations)-1)
		
		for gr in gold_instance.i_relations:
			for pr in pred_instance.i_relations:
				if gr.span[0][0] == pr.span[0][0] and gr.span[-1][-1] == pr.span[-1][-1]\
				and (gr.span[0][0] != 0 or gr.span[-1][-1] != gold_instance.segments_span[-1][-1]):
					self.parse_eval_data_dict['merge']['tp'] += 1
					if gr.sense == pr.sense:
						self.parse_eval_data_dict['sense']['tp'] += 1
					if gr.center == pr.center:
						self.parse_eval_data_dict['center']['tp'] += 1
					if gr.sense == pr.sense and gr.center == pr.center:
						self.parse_eval_data_dict['join']['tp'] += 1
		
	
	def show_vocabulary(self, word_to_ix):
		for word in word_to_ix:
			print word.encode('utf-8'), ':', word_to_ix[word], '\t',
	
	#set the start time, for measure process time
	def set_time(self):
		self.time = time.time()
	
	#log the process time
	def measure_time(self):
		end_time = time.time()
		process_time = end_time - self.time
		print('process time: %s\n'% process_time)
		sys.stderr.write('process time: %s\n'% process_time)

reporter = Reporter()