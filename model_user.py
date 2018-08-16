import data_structure
import rvnn
import evaluate

import sys
import copy
import time
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#whether to reduce instances by sampling
sample_flag = True

SAVE_MODEL_PATH = './model_test_loss_batch10_1000_10'

#for NN model config
EMBEDDING_DIM = 64
RVNN_HIDDEN_DIM = 64
BILSTM_CRF_HIDDEN_DIM = 64
LSTM_HIDDEN_DIM = 64
LEARNING_RATE = 0.1
SAMPLE_NUMBER = 30
#multiple factor of the segment/DU level instances to sample word level instances
SAMPLE_FACTOR = 1
EPOCHS = 10
BATCH_SIZE = 10
GPU = True

lstm_crf_only_flag = False
lstm_crf_flag = False


class Config():
	def __init__(self, vocab_size, d_embedding, d_bilstm_crf_hidden, tag_size, d_lstm_hidden, d_rvhidden, d_struct, d_center, d_relation, gpu):	
		self.vocab_size = vocab_size
		self.d_embedding = d_embedding
		self.d_bilstm_crf_hidden = d_bilstm_crf_hidden
		self.tag_size = tag_size
		self.d_lstm_hidden = d_lstm_hidden
		self.d_rvhidden = d_rvhidden
		self.d_struct = d_struct
		self.d_center = d_center
		self.d_relation = d_relation
		self.gpu = gpu

def get_model(model_path=None):
	if model_path != None:
		model = torch.load(model_path)
	elif lstm_crf_only_flag:
		model = rvnn.BiLSTM_CRF(len(data_structure.word_to_ix), max(data_structure.SEQ_TAG_TO_LABEL.values())+1, EMBEDDING_DIM, EMBEDDING_DIM)
	else:
		config = Config(len(data_structure.word_to_ix), EMBEDDING_DIM, BILSTM_CRF_HIDDEN_DIM, max(data_structure.SEQ_TAG_TO_LABEL.values())+1, LSTM_HIDDEN_DIM, RVNN_HIDDEN_DIM, data_structure.STRUCT_LABEL_DIM, data_structure.CENTER_DIM, data_structure.SENSE_DIM, GPU)
		model = rvnn.RVNN(config)

	#gpu
	if GPU:
		model.cuda()

	return model

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)	

def demo_predict(model, instance):
	instance.fragments = text_to_nn_word_list(instance.fragments)
	return_instance = model(instance)
	return return_instance
	
def train_from_instances(model, train_instances):
	
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	loss_function = nn.NLLLoss()
	if GPU:
		loss_function.cuda()
	for epoch in xrange(EPOCHS):
		start_time = time.time()
		#if sample_flag:
		#	train_instances = sample_train_instances(train_instances)
		random.shuffle(instances)
		
		instance_count = 0
		
		struct_loss_sum = 0
		center_loss_sum = 0
		sense_loss_sum = 0
		tag_loss_sum = 0
		objective_loss_sum = 0
		
		objective_loss = 0

		for train_instance in train_instances:
			#if train_instance.id[0:18] != '593.xml-5-train-27-13737'[0:18]:
			#	continue
			#data_structure.print_train_instance(train_instance)		
			if instance_count%100 == 0:
				print('trained instances: %d'%instance_count)
				sys.stderr.write('trained instances: %d'%instance_count)
			print('instance id: %s\r'%train_instance.id)
			sys.stderr.write('instance id: %s\r'%train_instance.id)
			instance_count += 1	
			#instance = train_instance
			instance = copy.deepcopy(train_instance)

			instance.fragments = text_to_nn_word_list(instance.fragments)
			model.zero_grad()
			'''
			if lstm_crf_only_flag:
				for words in train_instance.words_list:
					tag_labels = []				
					for word in words:
						tag_labels.extend(data_structure.word_to_labels(word))
					#tag_targets = torch.LongTensor(tags)
					#print tags
					tag_targets = prepare_tensor_train_ans(tag_labels)
					tag_loss = model.neg_log_likelihood(instance.fragments, tag_targets)
					#tag_loss = autograd.Variable(torch.LongTensor([0]))
					tag_loss_sum += tag_loss.data[0]
					objective_loss += tag_loss
			'''
			#else:
			#scores_list: [<tag_score_list>, struct_scores, <center_scores, sense_scores>]
			if instance.label == data_structure.STRUCT_LABEL_TRUE:
				instance, scores_list = model.forward_by_i_relations(instance, struct=True, center=True, sense=True)
			else:
				instance, scores_list = model.forward_by_i_relations(instance, struct=True, center=False, sense=False)
		
			score_idx = 0
				
			#for char tag
			if lstm_crf_flag:
				tag_score_list = scores_list[score_idx]
				tag_score_idx = 0
				instance1 = copy.deepcopy(train_instance)
				start_idx = instance1.segment_spans[0][0]
				for seg_idx in range(len(instance1.segment_spans)):
					#for gold tags
					tag_labels = []
					words = instance1.words_list[seg_idx]
					for word in words:
						tag_labels.extend(data_structure.word_to_labels(word))
					#for segment input
					span = instance1.segment_spans[seg_idx]
					sentence = text_to_nn_word_list(instance.fragments[span[0]-start_idx:span[-1]+1-start_idx])

					tag_targets = prepare_tensor_train_ans(tag_labels)
					gold_tag_score = model.score_sentence(sentence, tag_targets)
					#according to the lstm_crf implementation in pytorch tutorial, this subtraction computes NLLLoss
					tag_loss = tag_score_list[tag_score_idx] - gold_tag_score
					tag_score_idx  += 1
					tag_loss_sum += tag_loss.data[0]
					objective_loss += tag_loss
				score_idx += 1
				
			if not(lstm_crf_flag and lstm_crf_only_flag):
			
				#for struct
				struct_target = prepare_train_ans([instance.label])
				struct_loss = loss_function(scores_list[score_idx], struct_target)
				#print('struct_scores: ', scores_list[score_idx], 'struct_target: ', struct_target)
				struct_loss_sum += struct_loss.data[0]	
				objective_loss += struct_loss
				score_idx += 1
					
				if instance.label == data_structure.STRUCT_LABEL_TRUE and instance.segment_flag:
					#for center and sense
					#we use index -1 since the i_relations is already in post order
					center_target = prepare_train_ans([instance.i_relations[-1].center])
					center_loss = loss_function(scores_list[score_idx], center_target)
					center_loss_sum += center_loss
					score_idx += 1
					#print('center_scores: ', scores_list[score_idx], 'center_target: ', center_target)
					#we use index -1 since the i_relations is already in post order
					sense_target = prepare_train_ans([instance.i_relations[-1].sense])
					sense_loss = loss_function(scores_list[score_idx], sense_target)
					sense_loss_sum += sense_loss
					#print('sense_scores: ', scores_list[score_idx], 'sense_target: ', sense_target)
					objective_loss += center_loss
					objective_loss += sense_loss
			
			if instance_count%BATCH_SIZE == 0:
				objective_loss_sum += objective_loss.data[0]
				objective_loss.backward()
				optimizer.step()
				objective_loss = 0
		end_time = time.time()
		print('use time: %d'%(end_time-start_time))
		sys.stderr.write('use time: %d'%(end_time-start_time))
		print('epoch %d lose: struct: %f, center: %f, sense: %f, tag: %f, objective: %f\n'%(epoch, struct_loss_sum, center_loss_sum, sense_loss_sum, tag_loss_sum, objective_loss_sum))
		sys.stderr.write('epoch %d lose: struct: %f, center: %f, sense: %f, tag: %f, objective: %f\n'%(epoch, struct_loss_sum, center_loss_sum, sense_loss_sum, tag_loss_sum, objective_loss_sum))


		if SAVE_MODEL_PATH != None:
			torch.save(model,SAVE_MODEL_PATH)
	
	return

def test_from_corpus_list(model, corpus_list):
	evaluater = evaluate.Evaluater()
	evaluater2 = evaluate.Evaluater()
	evaluater3 = evaluate.Evaluater()
	count = 0
	print 'vocab_size', len(data_structure.word_to_ix)
	for corpus in corpus_list:
		
		#if count == int(sys.argv[1]):
		#	continue
		#print corpus.id
		#count += 1
		test_instance = data_structure.corpus_to_test_instance(corpus, binary=True)
		# multinuclear gold instnace
		gold_multi_instance = data_structure.corpus_to_test_instance(corpus, binary=False)
		gold_binary_instance = data_structure.corpus_to_test_instance(corpus, binary=True)
		#print test_instance.fragments

		test_instance.fragments = text_to_nn_word_list(test_instance.fragments)		
		model.zero_grad()
		#print 'test_instance'
		#data_structure.print_test_instance(test_instance)
		if lstm_crf_only_flag:
			_, labels = model(test_instance.fragments)
			#return_instance = data_structure.labels_to_words_in_test_instance(labels, instance)
		
		else:			
			return_instance = model(test_instance)
		#print 'corpus',
		#data_structure.print_corpus(corpus)
		return_instance.i_relations = data_structure.relations_to_post_order(return_instance.i_relations)
		'''
		print 'return_instance'
		data_structure.print_test_instance(return_instance)
		print 'gold_binary_instance'
		data_structure.print_test_instance(gold_binary_instance)
		print 'gold_multi_instance'
		data_structure.print_test_instance(gold_multi_instance)
		'''
		
		evaluater_tmp1 = evaluate.Evaluater()
		evaluater_tmp1.collect_eval_data(gold_binary_instance, return_instance)
		evaluater_tmp1.show_single_eval_result(gold_binary_instance, return_instance)
		
		evaluater.collect_eval_data(gold_binary_instance, return_instance)
		evaluater2.collect_eval_data(gold_multi_instance, return_instance)
		return_instance.du_i_relations = \
		 data_structure.relations_binary_to_multi_preorder(return_instance.du_i_relations)
		evaluater3.collect_eval_data(gold_multi_instance, return_instance)
		
		evaluater_tmp2 = evaluate.Evaluater()
		evaluater_tmp2.collect_eval_data(gold_multi_instance, return_instance)
		evaluater_tmp2.show_single_eval_result(gold_multi_instance, return_instance)
	
	evaluater.show_eval_result()
	evaluater2.show_eval_result()
	evaluater3.show_eval_result()
	return

def sample_train_instances(instances):
	
	pos_seg_du_instances = []
	pos_w_instances = []
	neg_seg_du_instances = []
	neg_w_instances = []

	for instance in instances:
		if instance.segment_flag:
			if instance.label == data_structure.STRUCT_LABEL_TRUE:
				pos_seg_du_instances.append(instance)
			else:
				neg_seg_du_instances.append(instance)
		else:
			if instance.label == data_structure.STRUCT_LABEL_TRUE:
				pos_w_instances.append(instance)
			else:
				neg_w_instances.append(instance)
	
	pos_seg_du_count = len(pos_seg_du_instances)
	neg_seg_du_count = len(neg_seg_du_instances)
	pos_w_count = len(pos_w_instances)
	neg_w_count = len(neg_w_instances)
	w_count = pos_w_count+neg_w_count
	seg_du_count = pos_seg_du_count + neg_seg_du_count
	
	
	if  w_count > SAMPLE_FACTOR*seg_du_count:
		w_instances = random.sample(neg_w_instances+pos_w_instances, SAMPLE_FACTOR*seg_du_count)
	else:
		w_instances = neg_w_instances+pos_w_instances
	
	instances = pos_seg_du_instances+neg_seg_du_instances+w_instances
	#shuffle
	random.shuffle(instances)
	#rename instance id
	count = 1
	for instance in instances:
		length = len(instance.id.split('-')[-1])
		instance.id = instance.id[0:-1*length]
		instance.id += str(count)
		count += 1
		
	return instances 

def print_loss(epoch, tag_loss_sum, struct_loss_sum, sense_loss_sum, center_loss_sum, objective_loss_sum):
	if lstm_crf_only_flag:
		print('epoch %d lose: tag: %f, objective: %f\n'%(epoch, tag_loss_sum.data[0], objective_loss_sum.data[0]))
		sys.stderr.write('epoch %d lose:tag: %f, objective: %f\n'%(epoch, tag_loss_sum.data[0], objective_loss_sum.data[0]))			
	else:
		print('epoch %d lose: struct: %f, center: %f, sense: %f, tag: %f, objective: %f\n'%(epoch, struct_loss_sum.data[0], center_loss_sum.data[0], sense_loss_sum.data[0], tag_loss_sum.data[0], objective_loss_sum.data[0]))
		sys.stderr.write('epoch %d lose: struct: %f, center: %f, sense: %f, tag: %f, objective: %f\n'%(epoch, struct_loss_sum.data[0], center_loss_sum.data[0], sense_loss_sum.data[0], tag_loss_sum.data[0], objective_loss_sum.data[0]))
	return 

	#four categories: whether they are above segment level or they are positive
def print_instnaces_categories(instances):
	pos_seg_du_instances = []
	pos_w_instances = []
	neg_seg_du_instances = []
	neg_w_instances = []

	for instance in instances:
		if instance.segment_flag:
			if instance.label == data_structure.STRUCT_LABEL_TRUE:
				pos_seg_du_instances.append(instance)
			else:
				neg_seg_du_instances.append(instance)
		else:
			if instance.label == data_structure.STRUCT_LABEL_TRUE:
				pos_w_instances.append(instance)
			else:
				neg_w_instances.append(instance)
	
	pos_seg_du_count = len(pos_seg_du_instances)
	neg_seg_du_count = len(neg_seg_du_instances)
	pos_w_count = len(pos_w_instances)
	neg_w_count = len(neg_w_instances)
	
	print pos_seg_du_count, neg_seg_du_count, pos_w_count, neg_w_count
	
def text_to_nn_word_list(text):
	#for t in text:
	#	print t.encode('utf-8'),
	#print ''
	idxs = map(lambda w: data_structure.word_to_ix.setdefault(w,data_structure.word_to_ix['oov']-1), text)
	tensor = torch.LongTensor(idxs)	
	if GPU:
		return autograd.Variable(tensor).cuda()
	else:
		return autograd.Variable(tensor) 

#for returning tensor, not variable
def prepare_tensor_train_ans(idxs):
	tensor = torch.LongTensor(idxs)
	#print tensor
	if GPU:
		return tensor.cuda()
	else:
		return tensor
		
def prepare_train_ans(idxs):
	tensor = torch.LongTensor(idxs)
	#print tensor
	if GPU:
		return autograd.Variable(tensor).cuda()
	else:
		return autograd.Variable(tensor)
	


	