# coding=UTF-8
from __future__ import print_function
from collections import defaultdict
import sys
import copy

import corpus
import args
import report
import lstm_rvnn

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




PRE_EDU_SENSE = 4
EDU_SENSE = 5
NON_SENSE = 6
NON_CENTER = 4


SENSE_TO_LABEL = {u'并列关系': 0, u'顺承关系': 0, u'递进关系':0, u'选择关系': 0, u'对比关系':0, u'因果关系': 1, u'推断关系': 1, u'假设关系': 1, u'目的关系': 1, u'条件关系': 1, u'背景关系': 1, u'转折关系': 2, u'让步关系': 2, u'解说关系': 3, u'总分关系': 3, u'例证关系': 3, u'评价关系': 3, PRE_EDU_SENSE: PRE_EDU_SENSE, EDU_SENSE: EDU_SENSE, NON_SENSE: NON_SENSE}

CENTER_TO_LABEL = {'1':0, '2':1, '3':2, NON_CENTER: NON_CENTER}

STRUCT_LABEL_TRUE = 1
STRUCT_LABEL_FALSE = 0

RELATION_LABEL_LIST = ['coordination', 'causality', 'transition', 'explanation']

#for NN model config
EMBEDDING_DIM = 64
WORD_HIDDEN_DIM = 64
SEG_HIDDEN_DIM = 32
RVNN_HIDDEN_DIM =64
LABEL_DIM = 2
CENTER_DIM = 5
RELATION_DIM = 7

# the function is to print stderr message
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


		

def get_model_lstm_rvnn(model_path=None):	
	
	if model_path != None:
		model = torch.load(model_path)

	else:
		config = lstm_rvnn.Config(RVNN_HIDDEN_DIM, WORD_HIDDEN_DIM, SEG_HIDDEN_DIM, len(corpus.word_to_ix), EMBEDDING_DIM, LABEL_DIM, CENTER_DIM, RELATION_DIM, args.args.gpu)
		model = lstm_rvnn.LSTM_RVNN(config)

	#gpu
	if args.args.gpu:
		model.cuda()

	return model
		
count = 0	
def train(model, corpus_list):
	
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	loss_function = nn.NLLLoss()
	if args.args.gpu:
		loss_function.cuda()
	
	train_instances = []
	#count= 0 
	for corpus in corpus_list:
		instances = get_train_instances_from_corpus(corpus)
		train_instances = train_instances + instances
		if args.args.one_train:
			train_instances = [train_instances[0]]
			break
		#if count == 5:
		#	break
		#count +=1
	
	for epoch in xrange(20):
		report.reporter.set_time()
	
		struct_loss_sum = 0
		center_loss_sum = 0
		sense_loss_sum = 0
		objective_loss_sum = 0
		
		global count
		count = 0
		
		for train_instance in train_instances:
			
			count += 1
			#print(count)
			#if count != 9835: continue

			#report.reporter.show_attributes(train_instance, object_attr='i_relations')
			instance = copy.deepcopy(train_instance)
			
			segments = []
			for seg_text in instance.segments:
				segment = text_to_nn_word_list(seg_text)
				segments.append(segment)
				instance.segments = segments
	
			model.zero_grad()

			#scores_list: [struct_scores <, center_scores, sense_scores>]
			if instance.label == STRUCT_LABEL_TRUE:
				scores_list = model.forward_train(instance, center=True, sense=True)
				pass
			else:
				scores_list = model.forward_train(instance)
	
			struct_target = prepare_train_ans(instance.label)
			struct_loss = loss_function(scores_list[0], struct_target)
			#print('struct_scores: ', scores_list[0], 'struct_target: ', struct_target)
			struct_loss_sum += struct_loss	
			
			objective_loss = struct_loss
			if instance.label == STRUCT_LABEL_TRUE:
				#we use index -1 since the i_relations is already in post order
				center_target = prepare_train_ans(instance.i_relations[-1].center)
				center_loss = loss_function(scores_list[1], center_target)
				center_loss_sum += center_loss
				#print('center_scores: ', scores_list[1], 'center_target: ', center_target)
				#we use index -1 since the i_relations is already in post order
				sense_target = prepare_train_ans(instance.i_relations[-1].sense)
				sense_loss = loss_function(scores_list[2], sense_target)
				sense_loss_sum += sense_loss
				#print('sense_scores: ', scores_list[2], 'sense_target: ', sense_target)
				objective_loss += center_loss
				objective_loss += sense_loss
			
			objective_loss_sum += objective_loss
			objective_loss.backward()
			optimizer.step()
		
		report.reporter.measure_time()
		print('epoch %d lose: struct: %f, center: %f, sense: %f, objective: %f'%(epoch, struct_loss_sum.data[0], center_loss_sum.data[0], sense_loss_sum.data[0], objective_loss_sum.data[0]))
		eprint('epoch %d lose: struct: %f, center: %f, sense: %f, objective: %f'%(epoch, struct_loss_sum.data[0], center_loss_sum.data[0], sense_loss_sum.data[0], objective_loss_sum.data[0]))
	
	if args.args.save_model != None:
		torch.save(model,args.args.save_model)
	
	return

def test(model, corpus_list):
	
	count = 0
	report.reporter.set_time()
	for corpus in corpus_list:
		test_instance = get_test_instance_from_corpus(corpus)
		# multinuclear gold instnace
		gold_instance = get_test_instance_from_corpus(corpus, gold=True)
		
		report.reporter.show_attributes(gold_instance, object_attr='i_relations')
		#report.reporter.show_attributes(test_instance, object_attr='i_relations')
		
		segments = []
		for seg_text in test_instance.segments:
			segment = text_to_nn_word_list(seg_text)
			segments.append(segment)
		test_instance.segments = segments
		
		model.zero_grad()
		return_instance = model(test_instance)
		report.reporter.collect_eval_data(gold_instance, return_instance)
		
		if args.args.one_test:
			break
		if count == 5:
			#break
			pass
		count += 1
	report.reporter.measure_time()
	report.reporter.show_eval_result()
		
	return

def model_train(model, optimizer, loss_function, instance, target, center=False, sense=False):
	segments = []
	for seg_text in instance.segments:
		segment = text_to_nn_word_list(seg_text)
		segments.append(segment)
	instance.segments = segments
	
	model.zero_grad()
	model.hidden = model.lstmmodel.init_hidden(args.args.gpu)
	if count == 9835:
		label_score = model.forward_train(instance, center=center, sense=sense, flag=False)
	else:
		label_score = model.forward_train(instance, center=center, sense=sense)
	'''
	if not sense and not center:
		print('struct score: ', label_score)
		print('struct target: ', target)
	if sense:
		print('sense score: ', label_score)
		print('sense target: ', target)
	'''
	loss = loss_function(label_score, target)
	
	#loss.backward()
	#optimizer.step()
	return loss

def prepare_train_ans(idx):
	idx = [idx]
	tensor = torch.LongTensor(idx)
	#print tensor
	if args.args.gpu:
		return autograd.Variable(tensor).cuda()
	else:
		return autograd.Variable(tensor)
	



def get_train_instances_from_corpus(in_corpus):
	
	#use a copy of the corpus to convert to binary structure, not the original object
	
	in_corpus = copy.deepcopy(in_corpus)
	in_corpus.to_binary_structure()
	
	#report.reporter.show_attributes(in_corpus, object_attr='relations')
	
	instances = []
	
	# for finding pairs, we only store the span, not the whole Relation object
	#after find all neighboring pairs, we recover the relation structure and make instances

	edu_extended_relations = extend_edu_relation(in_corpus)
	edu_extended_all_relations = edu_extended_relations + in_corpus.relations
	
	# this list is the main use of the following
	relations = edu_extended_all_relations
	relations = relations_to_pre_order(relations)
	#the key of the span_to_relation_structure_dict is a span, ex: [ [0,5],[6,10] ]
	#the value is the pre order relations when see the key span as root
	span_to_relation_structure_dict = get_span_to_relation_structure_dict(relations)

	
	#span of all nodes including edu(, segments)
	span_list = [ relation.span for relation in relations ]
	
	#this may be used when edu base
	#span_list = [ [span] for span in in_corpus.edus_span ] + span_list
	span_list = [ [span] for span in in_corpus.segments_span ] + span_list
	
	#sort the new list to pre order
	span_list = spans_to_pre_order(span_list)

	
	
	for i in range(len(span_list)):
		now_span = span_list[i]
		#thanks for the pre order, we only consider the spans after now_span
		for j in range(i, len(span_list)):
			# if the span is neighboring right to the now_span, then we can get an instance  
			if span_list[j][0][0] == now_span[-1][-1]+1:
				recovered_relations = recover_relations_from_span_pair(now_span, span_list[j], span_to_relation_structure_dict)
				instance = get_instance_from_corpus_and_relations(in_corpus, recovered_relations)
				#to post order, for construct structure when training
				instance.i_relations = i_relations_to_post_order(instance.i_relations)
				#report.reporter.show_attributes(instance, object_attr='i_relations')
				instances.append(instance)
	
	return instances

def get_test_instance_from_corpus(corpus, gold=False):
	corpus = copy.deepcopy(corpus)
	
	if gold:
		corpus.to_binary_structure()
	
	instance = lstm_rvnn.Instance()
	
	if args.args.gold_edu_direct:
		instance.segments_span = corpus.edus_span
	else:	
		instance.segments_span = corpus.segments_span
		
	
	#for segments and puncs
	#ex : [0,5]
	for span in instance.segments_span:
		text = corpus.text[ span[0]: span[-1]+1 ] 
		segment = text 
		instance.segments.append( segment )
		instance.puncs.append(segment[-1])
	
	# for pre_edu relation
	edu_extended_relations = extend_edu_relation(corpus)
	#edu_extended_all_relations = edu_extended_relations + corpus.relations
	
	instance.i_relations = []
	
	if gold:
		relations = corpus.relations
	elif args.args.gold_edu:
		#only for pre edu i_relations
		relations = edu_extended_relations
	else:
		relations = []
	
	for relation in relations:
		i_relation = lstm_rvnn.I_Relation(relation.span, SENSE_TO_LABEL[relation.sense], CENTER_TO_LABEL[relation.center])
		instance.i_relations.append(i_relation)
	
	#to post order, for construct edu when predicting
	instance.i_relations = i_relations_to_post_order(instance.i_relations)
	
	return instance
	
def extend_edu_relation(in_corpus):
	edu_extended_relations = []
	
	seg_i = 0
	for edu_span in in_corpus.edus_span:
		#right first merge
		if args.args.train_right_edu:
			right_end = edu_span[-1]
			left_end = edu_span[0]
			left_span = []
			while in_corpus.segments_span[seg_i][-1] <= right_end:
				left_span = in_corpus.segments_span[seg_i]
				if seg_i + 1 == len(in_corpus.segments_span):
					break
				elif left_span[-1] < right_end:
					if left_span[0] == left_end:
						sense = EDU_SENSE
					else:
						sense = PRE_EDU_SENSE
					relation = corpus.Relation([ left_span,[ left_span[-1]+1,right_end ] ], sense, NON_CENTER)
					#report.reporter.show_attributes(relation)
					edu_extended_relations.append(relation)
				seg_i += 1
		else:
			#initial blank left node
			left_span = []
			#while still in the edu
			while in_corpus.segments_span[seg_i][-1] <= edu_span[-1]:
				if left_span == []:
					left_span = in_corpus.segments_span[seg_i]
				else:
					right_span = in_corpus.segments_span[seg_i]
					# a edu is finished
					if right_span[-1] == edu_span[-1]:
						sense = EDU_SENSE
					else:
						sense = PRE_EDU_SENSE
					relation = corpus.Relation([left_span,right_span], sense, NON_CENTER)
					#report.reporter.show_attributes(relation)
					edu_extended_relations.append(relation)
					# make a new left node span
					left_span = [ left_span[0], right_span[-1] ]
				# if it is the last segment
				if seg_i + 1 == len(in_corpus.segments_span):
					break
				
				seg_i += 1
	
	return edu_extended_relations

#the key of the span_to_relation_structure_dict is a span, ex: [ [0,5],[6,10] ]
#the value is the pre order relations when see the key span as root
def get_span_to_relation_structure_dict(relations):
	
	#[] when use edu span as key in the future
	span_to_relation_structure_dict = defaultdict(list)
	
	
	#build span_to_relation_structure_dict
	for i in range(len(relations)):
		now_span = relations[i].span
		#thanks for the pre order, we only consider the spans 'now' and after now_span
		for j in range(i, len(relations)):
			# if the span is out of now_span
			if relations[j].span[0][0] < now_span[0][0] or relations[j].span[-1][-1] > now_span[-1][-1]: break
			#use str() to make list hashable
			span_to_relation_structure_dict[str(now_span)].append( relations[j] )

	return span_to_relation_structure_dict
	
def recover_relations_from_span_pair(span_left, span_right, span_to_relation_structure_dict):
	instace = lstm_rvnn.Instance()
	
	#get the start boundary and end boundary of span_left and span_right 
	span_unit_left = [ span_left[0][0], span_left[-1][-1] ]
	span_unit_right = [ span_right[0][0], span_right[-1][-1] ]
	
	root_span = [ span_unit_left, span_unit_right ]
	
	relation_list = span_to_relation_structure_dict[str(root_span)]
	
	# if in the dict, the root span doesn't exist, it means the relation is not true in the original structure.We construct a psuedo Relation for root relation where sense and center is None
	if relation_list == []:
		relation_list = span_to_relation_structure_dict[str(span_left)] + span_to_relation_structure_dict[str(span_right)]
		root_relation = corpus.Relation(root_span, NON_SENSE, NON_CENTER)
		relation_list = [root_relation] + relation_list

	return relation_list
	
def get_instance_from_corpus_and_relations(corpus, relations):
	
	instance = lstm_rvnn.Instance()
	
	i_relations = []
	for relation in relations:
		i_relation = lstm_rvnn.I_Relation(relation.span, SENSE_TO_LABEL[relation.sense], CENTER_TO_LABEL[relation.center] )
		i_relations.append(i_relation)	
	instance.i_relations = i_relations	
	
	if i_relations[0].sense == NON_SENSE:
		instance.label = STRUCT_LABEL_FALSE
	else:
		instance.label = STRUCT_LABEL_TRUE
	
	start_idx = i_relations[0].span[0][0]
	end_idx = i_relations[0].span[-1][-1]
	'''
	global count
	count += 1
	if count == 9835:
		print (start_idx, end_idx)
		print (corpus.id)
	'''
	
	segments_span = []
	# find the range of the instance span in the corpus span 
	#for span in corpus.edus_span:
	for span in corpus.segments_span:
		if start_idx <= span[0] and end_idx >= span[-1]:
			segments_span.append(span)
		
	instance.segments_span = segments_span
	
	#for segments and puncs
	#ex : [0,5]
	for span in segments_span:
		text = corpus.text[ span[0]: span[-1]+1 ] 
		segment = text 
		instance.segments.append( segment )
		instance.puncs.append( segment[-1] )	
	
	return instance

def text_to_nn_word_list(text):

	idxs = map(lambda w: corpus.word_to_ix.setdefault(w,corpus.oov), text)
	tensor = torch.LongTensor(idxs)	
	if args.args.gpu:
		return autograd.Variable(tensor).cuda()
	else:
		return autograd.Variable(tensor) 

	#return None

def spans_to_pre_order(span_list):
	# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
	from functools import cmp_to_key
	#cmp_for_pre_order_from_span: compare function for sorting for pre order
	key_for_pre_order_from_span = cmp_to_key(cmp_for_pre_order_from_span)
		
	span_list = sorted(span_list, key=key_for_pre_order_from_span, reverse=True)
		
	return span_list

def i_relations_to_post_order(i_relations):
	# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
	from functools import cmp_to_key
	#cmp_for_post_order_from_i_relation: compare function for sorting for post order
	key_for_post_order_from_i_relation = cmp_to_key(cmp_for_post_order_from_i_relation)
		
	i_relations = sorted(i_relations, key=key_for_post_order_from_i_relation, reverse=True)
		
	return i_relations
	
#compare function for sorting for pre order
def cmp_for_pre_order_from_span(span_1, span_2):
	#compare the start boundary first
	if span_1[0][0] < span_2[0][0]:
		return 1
		#if same, compare the end boundary
	elif span_1[0][0] == span_2[0][0]:
		if span_1[-1][-1] > span_2[-1][-1]:
			return 1
	return -1

#compare function for sorting for post order
def cmp_for_post_order_from_i_relation(ir_1, ir_2):
	#compare the end boundary first
	if ir_1.span[-1][-1] < ir_2.span[-1][-1]:
		return 1
		#if same, compare the start boundary
	elif ir_1.span[-1][-1] == ir_2.span[-1][-1]:
		if ir_1.span[0][0] > ir_2.span[0][0]:
			return 1
	return -1

def relations_to_pre_order(relations):
	# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
	from functools import cmp_to_key
	#cmp_for_post_order_from_i_relation: compare function for sorting for post order
	key_for_pre_order_from_relation = cmp_to_key(cmp_for_pre_order_from_relation)
		
	relations = sorted(relations, key=key_for_pre_order_from_relation, reverse=True)
		
	return relations	

#compare function for sorting for pre order
def cmp_for_pre_order_from_relation(r_1, r_2):
	#compare the start boundary first
	if r_1.span[0][0] < r_2.span[0][0]:
		return 1
		#if same, compare the end boundary
	elif r_1.span[0][0] == r_2.span[0][0]:
		if r_1.span[-1][-1] > r_2.span[-1][-1]:
			return 1
	return -1
	