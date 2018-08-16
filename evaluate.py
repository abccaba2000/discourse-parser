import data_structure

from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

class Evaluater():
	def __init__(self):		
		#for du level parseval
		self.du_parse_eval_data_dict = {'merge': {}, 'sense': {}, 'center': {}, 'join': {}}
		#for word level parseval
		self.w_parse_eval_data_dict = {'merge': {}, 'sense': {}, 'center': {}, 'join': {}}
		#for char tags
		self.char_tags_eval_data_dict = {'word_seg': {}, 'pos': {}}
		for key in self.du_parse_eval_data_dict:
			self.du_parse_eval_data_dict[key] = {'tp':0, 'gold_n':0, 'pred_n':0}
			self.w_parse_eval_data_dict[key] = {'tp':0, 'gold_n':0, 'pred_n':0}
		for key in self.char_tags_eval_data_dict:	
			self.char_tags_eval_data_dict[key] = {'tp':0, 'gold_n':0, 'pred_n':0}
		#for edu
		self.gold_edu_tag_list = []
		self.pred_edu_tag_list = []

		#order: sense1, sense2, center, child_num, binary, explicit
		self.relation_distribution_eval_dict = defaultdict(lambda: defaultdict(int))
		self.edu_punc_distrbution = defaultdict(lambda: defaultdict(int))
		
	def collect_eval_data(self, gold_te_instance, pred_te_instance):		
		#self.get_tag_eval_data(gold_te_instance, pred_te_instance)
		self.get_edu_eval_data(gold_te_instance, pred_te_instance)
		self.get_du_parse_eval_data(gold_te_instance, pred_te_instance)

	def get_tag_eval_data(self, gold_te_instance, pred_te_instance):
		self.char_tags_eval_data_dict['word_seg']['gold_n'] += len(gold_te_instance.words)
		self.char_tags_eval_data_dict['word_seg']['pred_n'] += len(pred_te_instance.words)
		self.char_tags_eval_data_dict['pos']['gold_n'] += len(gold_te_instance.words)
		self.char_tags_eval_data_dict['pos']['pred_n'] += len(pred_te_instance.words)
		for gold_word in gold_te_instance.words:
			for pred_word in pred_te_instance.words:
				if gold.word.span == pred.word.span:
					self.char_tags_eval_data_dict['word_seg']['tp'] += 1
					#we need to use POS_TO_LABEL since different pos may belong to same class
					if data_structure.POS_TO_LABEL[gold.word.pos] == data_structure.POS_TO_LABEL[pred.word.pos]:
						self.char_tags_eval_data_dict['pos']['tp'] += 1
		
	def get_edu_eval_data(self, gold_te_instance, pred_te_instance):
		
		gold_edu_boundary_list = [span[-1] for span in gold_te_instance.edu_spans]
		pred_edu_spans = data_structure.relations_to_edu_spans(pred_te_instance.du_i_relations)
		pred_edu_boundary_list = [span[-1] for span in pred_edu_spans]
		
		
		gold_edu_tag_list = []
		pred_edu_tag_list = []
		punc_list = gold_te_instance.puncs
		
		
		seg_boundary_list = [ span[-1] for span in gold_te_instance.segment_spans ]
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
			print tag, punc.encode('utf-8'),
		print ''
		for tag, punc in zip(pred_edu_tag_list,punc_list):
			print tag, punc.encode('utf-8'),	
		print '\n\n'
		#print gold_edu_tag_list
		#print pred_edu_tag_list
		
		self.gold_edu_tag_list.extend(gold_edu_tag_list)
		self.pred_edu_tag_list.extend(pred_edu_tag_list)
		return
		
	def get_du_parse_eval_data(self, gold_te_instance, pred_te_instance):
	
		for key in self.du_parse_eval_data_dict:
			# use max() in case of only one edu
			self.du_parse_eval_data_dict[key]['gold_n'] += max(0,len(gold_te_instance.du_i_relations)-1)
			self.du_parse_eval_data_dict[key]['pred_n'] += max(0,len(pred_te_instance.du_i_relations)-1)
		
		for pr in pred_te_instance.du_i_relations:
			self.relation_distribution_eval_dict[str(pr.sense)+'_all']['pred_n'] += 1
			self.relation_distribution_eval_dict[pr.type+'_all']['pred_n'] += 1			
		for gr in gold_te_instance.du_i_relations:
			self.relation_distribution_eval_dict[str(gr.sense)+'_all']['gold_n'] += 1
			self.relation_distribution_eval_dict[gr.type+'_all']['gold_n'] += 1
		
		for pr in pred_te_instance.du_i_relations:
			for gr in gold_te_instance.du_i_relations:
				if gr.spans[0][0] == pr.spans[0][0] and gr.spans[-1][-1] == pr.spans[-1][-1]:
					self.relation_distribution_eval_dict[str(pr.sense)]['pred_n'] += 1
					self.relation_distribution_eval_dict[pr.type]['pred_n'] += 1
					self.relation_distribution_eval_dict[str(gr.sense)]['gold_n'] += 1
					self.relation_distribution_eval_dict[gr.type]['gold_n'] += 1						
					if gr.sense == pr.sense:
						self.relation_distribution_eval_dict[str(gr.sense)+'_all']['tp'] += 1
						self.relation_distribution_eval_dict[gr.type+'_all']['tp'] += 1		
						self.relation_distribution_eval_dict[str(gr.sense)]['tp'] += 1
						self.relation_distribution_eval_dict[gr.type]['tp'] += 1					
					# exclude the root node
					if gr.spans[0][0] != 0 or gr.spans[-1][-1] != gold_te_instance.segment_spans[-1][-1]:
						self.du_parse_eval_data_dict['merge']['tp'] += 1
						if gr.sense == pr.sense:
							self.du_parse_eval_data_dict['sense']['tp'] += 1
						if gr.center == pr.center:
							self.du_parse_eval_data_dict['center']['tp'] += 1
						if gr.sense == pr.sense and gr.center == pr.center:
							self.du_parse_eval_data_dict['join']['tp'] += 1

	def show_eval_result(self):
	
		edu_result = precision_recall_fscore_support(self.gold_edu_tag_list, self.pred_edu_tag_list, average='binary')
		print 'edu_result:', edu_result
		for key, dict in self.du_parse_eval_data_dict.iteritems():
			print key,': ',
			for k, v in dict.iteritems():
				print k, ': ', v,
			if dict['gold_n'] != 0:
				print 'recall: ', float( dict['tp'] )/dict['gold_n'],
			else:
				print 'recall: 0',
			if dict['pred_n'] != 0:
				print 'precision: ', float( dict['tp'] )/dict['pred_n'],
			else:
				print 'precision: 0',
			if dict['gold_n'] + dict['pred_n'] != 0:
				print 'f1: ', 2*float( dict['tp'] )/( dict['gold_n'] + dict['pred_n'] )
			else:
				print 'f1: 0' 
		
		for key, dict in self.relation_distribution_eval_dict.iteritems():
			if type(key) == type(1):
				print key,': ',
			else:
				print key.encode('utf-8'),': ',
			for k, v in dict.iteritems():
				print k, ': ', v,
			if dict['gold_n'] != 0:
				print 'recall: ', float( dict['tp'] )/dict['gold_n'],
			else:
				print 'recall: 0',
			if dict['pred_n'] != 0:
				print 'precision: ', float( dict['tp'] )/dict['pred_n'],
			else:
				print 'precision: 0',
			if dict['gold_n'] + dict['pred_n'] != 0:
				print 'f1: ', 2*float( dict['tp'] )/( dict['gold_n'] + dict['pred_n'] )
			else:
				print 'f1: 0' 

	def show_single_eval_result(self, gold_instance, pred_instance):
	
		edu_result = precision_recall_fscore_support(self.gold_edu_tag_list, self.pred_edu_tag_list, average='binary')
		f1_dict = {}
		for key, dict in self.du_parse_eval_data_dict.iteritems():
			print key,': ',
			if dict['gold_n'] + dict['pred_n'] != 0:
				f1_dict[key] = 2*float( dict['tp'] )/( dict['gold_n'] + dict['pred_n'] )
			else:
				f1_dict[key] = 0
		
		if f1_dict['join'] > -1.0 and f1_dict['join'] < 1.1:
			print edu_result
			for k, v in f1_dict.iteritems():
				print k, v 
			data_structure.print_test_instance(gold_instance)
			data_structure.print_test_instance(pred_instance)
		
			
	
	def show_relation_distribution_from_corpus_list(self, corpus_list):
		#order: sense1, sense2, center, child_num, binary, explicit
		relation_distribution_dict = defaultdict(int)
		example_count_dict = defaultdict(int)
		example_dict = defaultdict(list)

		for corpus in corpus_list:
			for relation in corpus.du_relations:
				key_s1 = str(data_structure.SENSE_TO_LABEL[relation.sense])
				key_s2 = relation.sense
				key_c = str(relation.center)
				key_child_num = str(len(relation.spans))
				if len(relation.spans) == 2:
					key_binary = 'binary'
				else:
					key_binary = 'multi'
				key_type = relation.type
				key_list = [key_s1,key_s2,key_c,key_child_num,key_binary, key_type]
				key_combinations = ['']
				for key in key_list:
					new_combinations = []
					for key_combination in key_combinations:					
						new_combinations.append(key_combination+'_'+key)
						new_combinations.append(key_combination+'_'+'N')
					key_combinations = new_combinations[:]
				for key_cb in key_combinations:
					relation_distribution_dict[key_cb] += 1
				if len(relation.spans) >= 7:
					print len(relation.spans)
					for span in relation.spans:
						print corpus.text[span[0]:span[-1]+1].encode('utf-8')					
				
				key1 = key_s1+'_'+key_type
				key2 = key_s1+'_'+key_c
				if example_count_dict[key1] < 3:
					example_dict[key1].append([])
					for span in relation.spans:
						example_dict[key1][-1].append(corpus.text[span[0]:span[-1]+1].encode('utf-8'))
					example_count_dict[key1] += 1
				if example_count_dict[key2] < 3:
					example_dict[key2].append([])
					for span in relation.spans:
						example_dict[key2][-1].append(corpus.text[span[0]:span[-1]+1].encode('utf-8'))
					example_count_dict[key2] += 1

		for key in example_dict:
			print key.encode('utf-8')
			for example in example_dict[key]:
				for argument in example:
					print argument
				print ''
				
		relation_distribution_list = sorted(relation_distribution_dict.iteritems(), key=lambda x:x[0] )
		for k_v in relation_distribution_list:
			print k_v[0].encode('utf-8')+':', k_v[1],',\t',

	def show_edu_punc_distribution_from_corpus_list(self, corpus_list):
		for corpus in corpus_list:
			for span in corpus.segment_spans:
				self.edu_punc_distrbution['punc_number'][corpus.text[span[-1]].encode('utf-8')] += 1
			for span in corpus.edu_spans:
				self.edu_punc_distrbution['edu_punc_number'][corpus.text[span[-1]].encode('utf-8')] += 1
		for key, v in self.edu_punc_distrbution['punc_number'].iteritems():
			edu_v = self.edu_punc_distrbution['edu_punc_number'][key]
			print key, v, edu_v, float(edu_v)/v
	
	def show_analysis_from_corpus_list(self, corpus_list):
		edu_n = 0
		paragraph_n = 0
		relation_n = 0
		for corpus in corpus_list:
			edu_n += len(corpus.edu_spans)
			paragraph_n += 1
			relation_n += len(corpus.du_relations)
		print 'edu_n:', edu_n
		print 'paragraph_n:', paragraph_n
		print 'relation_n:', relation_n

	def show_pos_distribution_from_corpus_list(self, corpus_list):
		pos_distribution = defaultdict(int)
		for corpus in corpus_list:
			for word in corpus.words:
				pos_distribution[word.pos] += 1
		print pos_distribution