# coding=UTF-8
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd

import args
import report

DU_SENSE = (0,1,2,3)
PRE_EDU_SENSE = 4
EDU_SENSE = 5
NON_SENSE = 6
NON_CENTER = 4

N_CKY_CANDIDATES = 2

if args.args.punctuation_limit_edu:
	EDU_PUNCS = (u'。')
	PRE_EDU_PUNCS = (u'、')
else:
	EDU_PUNCS = ()
	PRE_EDU_PUNCS = ()

class Instance():
	def __init__(self):
		#each element is words index of a segment text the NN model needs.
		#ex: [ [3,5,4]Variable, [11,2,8]Variable ] 
		self.segments = []
		#ex:[u'，', u'。']
		self.puncs = []
		#ex:[[0, 46], [47, 61], [62, 76]]
		self.segments_span = []
		self.i_relations = []
		#True or False
		self.label = ''

class I_Relation():
	def __init__(self, span, sense, center):
		# ex: [[0, 46], [47, 76]]
		self.span = span
		self.sense = sense
		self.center = center

class CKY_Unit():
	def __init__(self, cky_candidate_list):
		self.cky_candidate_list = cky_candidate_list
		

class CKY_Candidate():
	def __init__(self, cky_span_infos, representation, score, sense, center):
		#a list, each element contains "start_idx" and "range" and "candidate_idx", we have "candidate_idx" since although we find a candidate cky_unit, it has many candidates also
		self.cky_span_infos = cky_span_infos
		self.representation = representation
		self.score = score
		self.sense = sense
		self.center = center

class CKY_Span_Info():
	def __init__(self, start_idx, cky_range, candidate_idx):
		self.start_idx = start_idx
		self.cky_range = cky_range
		self.candidate_idx = candidate_idx
		
class Config():
	def __init__(self, d_rvhidden, d_word_lstmhidden, d_seg_lstmhidden, vocab_size, d_embedding, d_struct, d_center, d_relation, gpu):
		self.d_rvhidden = d_rvhidden
		self.d_word_lstmhidden = d_word_lstmhidden
		self.d_seg_lstmhidden = d_seg_lstmhidden
		self.vocab_size = vocab_size
		self.d_embedding = d_embedding
		self.d_struct = d_struct
		self.d_center = d_center
		self.d_relation = d_relation
		self.gpu = gpu

	
		
class LSTM_RVNN(nn.Module):
	def __init__(self, config):
		super(LSTM_RVNN, self).__init__()
	
		self.double_lstmmodel = Double_LSTMModel(config.d_embedding, config.d_word_lstmhidden, config.d_seg_lstmhidden, config.vocab_size, config.gpu)
		self.reduce = Reduce(config.d_rvhidden/2)
		self.struct_linear = nn.Linear(config.d_rvhidden, config.d_struct)
		self.center_linear = nn.Linear(config.d_rvhidden, config.d_center)
		self.sense_linear = nn.Linear(config.d_rvhidden, config.d_relation)
	
	def reconstruct_i_relations_from_cky_table(self, instance, cky_table, cky_candidate):
	
		infos = cky_candidate.cky_span_infos		
		#recursive to the bottom(or edu)
		if len(infos) == 1 or cky_candidate.sense == EDU_SENSE or cky_candidate.sense == PRE_EDU_SENSE:
			return
		
		left_start_idx = infos[0].start_idx
		left_end_idx = infos[0].start_idx+infos[0].cky_range
		left_span = [ instance.segments_span[left_start_idx][0], instance.segments_span[left_end_idx][-1] ]
		
		right_start_idx = infos[-1].start_idx
		right_end_idx = infos[-1].start_idx+infos[-1].cky_range
		right_span = [ instance.segments_span[right_start_idx][0], instance.segments_span[right_end_idx][-1] ]
		
		i_relation = I_Relation([left_span, right_span], cky_candidate.sense, cky_candidate.center)
		
		#modify the instance directly
		instance.i_relations.append(i_relation)
		
		#find next recursive target
		for info in infos:
			next_candidate = cky_table[info.start_idx][info.cky_range].cky_candidate_list[info.candidate_idx]
			self.reconstruct_i_relations_from_cky_table(instance, cky_table, next_candidate)
		return
	
	def forward_train(self, instance, center=False, sense=False, flag= False):
		#send the idx list to lstm to get the sentence embedding list
		if flag:
			report.reporter.show_attributes(instance, object_attr='i_relations')
		#print(instance.segments)
		instance.segments = list(self.double_lstmmodel(instance.segments))
		for i_relation in  instance.i_relations:
			#print i_relation.span
			#print instance.segments_span
			#print '\n'
			left_span = i_relation.span[0]
			for idx in range(len(instance.segments_span)):
				#since the i_relations is in post order(buttom up stlye), it's always to find a corresponding segment to merge
				if left_span == instance.segments_span[idx]:
					left = instance.segments[idx]
					right = instance.segments[idx+1]
					reduced = self.reduce(left, right)
					instance.segments[idx] = reduced
					del instance.segments[idx+1]
					#modify the segment span list to follow the change of segment list
					instance.segments_span[idx] = [ instance.segments_span[idx][0], instance.segments_span[idx+1][-1] ]
					del instance.segments_span[idx+1]
					#finish the merge, break to deal with next relation
					break
		final_representation = instance.segments[0]
		
		scores_list = []
		
		label_space = self.struct_linear(final_representation)
		struct_scores= F.log_softmax(label_space)	
		scores_list.append(struct_scores)
		
		if center:	
			label_space = self.center_linear(final_representation)
			center_scores = F.log_softmax(label_space)
			scores_list.append(center_scores)
		
		if sense:
			label_space = self.sense_linear(final_representation)
			sense_scores = F.log_softmax(label_space)
			scores_list.append(sense_scores)
				

		
		return scores_list
		
	def forward_process_gold_edu(self, instance):
		#there are only pre edu level i_relations
		for i_relation in  instance.i_relations:
			#print i_relation.span
			#print instance.segments_span
			#print '\n'
			left_span = i_relation.span[0]
			for idx in range(len(instance.segments_span)):
				#since the i_relations is in post order(buttom up stlye), it's always to find a corresponding segment to merge
				if left_span == instance.segments_span[idx]:
					left = instance.segments[idx]
					right = instance.segments[idx+1]
					reduced = self.reduce(left, right)
					instance.segments[idx] = reduced
					del instance.segments[idx+1]
					#modify the segment span list to follow the change of segment list
					instance.segments_span[idx] = [ instance.segments_span[idx][0], instance.segments_span[idx+1][-1] ]
					del instance.segments_span[idx+1]
					#finish the merge, break to deal with next relation
					break
		
		#empty the pre edu i_relations, for fill structure i_relation prediction in the future
		instance.i_relations = []
		
	def forward_sequence_predict_edu(self, instance):
	
		#pre_edustage
		n_segments = len(instance.segments)
		#use flag instead of use torch Variable itself since the comparison limit, ex: can't write "if (torch Variable) == 0:" 
		left_flag = False
		right_flag = False
		edus = []
		edus_span = []
		#from left to right
		for idx in range(n_segments):
			punc = instance.puncs[idx]
			#merge or just set the left segment
			if not left_flag:
				left = instance.segments[idx]
				left_span = instance.segments_span[idx]
				left_flag = True
			else:
				#merge, make sense
				right = instance.segments[idx]
				right_span = instance.segments_span[idx]
				right_flag = True
				reduced = self.reduce(left, right)
				merge_span = [left_span[0], right_span[1]]	
				label_space = self.sense_linear(reduced)
				label_score = F.log_softmax(label_space)
				sense_score = [ x.data[0] for x in label_score[0]  ]
				merge_sense = sense_score.index( max(sense_score) )
			
			#modify the instance information, no need to deal with the i_reltion
			if right_flag:
				if merge_sense == EDU_SENSE and punc not in PRE_EDU_PUNCS:
					edus.append(reduced)
					edus_span.append(merge_span)
					left_flag = False
				else:
					if merge_sense == PRE_EDU_SENSE or punc in PRE_EDU_PUNCS:
						left = reduced
						left_span = merge_span
					else:
						edus.append(left)
						edus_span.append(left_span)
						left = right
						left_span = right_span
					#left is already modified above
					if idx == n_segments-1 or punc in EDU_PUNCS:
						edus.append(left)
						edus_span.append(left_span)
						left_flag = False
			else:
				if idx == n_segments-1 or punc in EDU_PUNCS:
					edus.append(left)
					edus_span.append(left_span)
					left_flag = False						
				
			right_flag = False
			
		# set the edus, modify the instance segments to edu level
		instance.segments = edus
		instance.segments_span = edus_span
	
	def forward_greedy_predict_structure(self, instance):
		i_relations = []
		while len(instance.segments) > 1:
			n_segments = len(instance.segments)
			scores = []
			reduceds = []
			for i in range(n_segments-1):
				reduced = self.reduce(instance.segments[i], instance.segments[i+1])
				reduceds.append(reduced)
				label_space = self.struct_linear(reduced)
				label_score = F.log_softmax(label_space)
				scores.append(label_score[0][1].data[0])
			#for span
			idx = scores.index( max(scores) )
			left_span = instance.segments_span[idx]
			right_span = instance.segments_span[idx+1]
			i_relation_span = [left_span, right_span]
			print 'left_span: ', left_span, 'right_span: ', right_span, 'max score: ', max(scores)
			#for center
			label_space = self.center_linear(reduceds[idx])
			label_score = F.log_softmax(label_space)
			center_score = [ x.data[0] for x in label_score[0]  ]
			#exclude edu centers
			center_score = center_score[0:3]
			idx_center = center_score.index( max(center_score) )

			#for sense
			label_space = self.sense_linear(reduceds[idx])
			label_score = F.log_softmax(label_space)
			sense_score = [ x.data[0] for x in label_score[0]  ]
			#exclude edu senses
			sense_score = sense_score[0:4]
			idx_sense = sense_score.index( max(sense_score) )
				
			#modify the instance information
			instance.segments[idx] = reduceds[idx]
			instance.segments_span[idx] = [ i_relation_span[0][0], i_relation_span[-1][-1] ]
			del instance.segments[idx+1]
			del instance.segments_span[idx+1]
			#make a new i_relation
			i_relation = I_Relation(i_relation_span, idx_sense, idx_center)
			i_relations.append(i_relation)
		instance.i_relations = i_relations
	
	#not performance well since if we don't construct EDU from the left most of a gold EDU, it's likely to be wrong(because of how we train the model)
	def forward_greedy_predict_edu(self,instance):
		n_segments = len(instance.segments)
		#to trace the preEDU/EDU status of the segments
		segment_sense_list = [PRE_EDU_SENSE]*n_segments
		last_n_segments = 0
		puncs = instance.puncs[:]
		#check if no other segments to merge, or no legal merge in the last iteration
		while n_segments > 0 and (n_segments != last_n_segments):
			last_n_segments = n_segments
			#each element is a tuple, containing the segment index(left) and the score
			#ex: (1, 0.22)
			idx_scores = []
			# need to be index by the segments index
			reduceds = [None]*n_segments
			for i in range(n_segments-1):
				#check whether some segment is already EDU 
				if segment_sense_list[i] == PRE_EDU_SENSE and segment_sense_list[i+1] == PRE_EDU_SENSE and puncs[i] not in EDU_PUNCS:
					reduced = self.reduce(instance.segments[i], instance.segments[i+1])
					reduceds[i] = reduced
					label_space = self.struct_linear(reduced)
					label_score = F.log_softmax(label_space)
					idx_scores.append( (i, label_score[0][1].data[0]) )
			#sort and try to merge from the highest score
			sorted_idx_scores = sorted(idx_scores, key=lambda x: x[1], reverse=True)
			#print segment_sense_list
			#print sorted_idx_scores
			for idx_score in sorted_idx_scores:
				idx = idx_score[0]
				#print idx,
				left_span = instance.segments_span[idx]
				right_span = instance.segments_span[idx+1]
				merge_span = [left_span[0], right_span[-1]]
				#for sense
				label_space = self.sense_linear(reduceds[idx])
				label_score = F.log_softmax(label_space)
				sense_score = [ x.data[0] for x in label_score[0]  ]
				idx_sense = sense_score.index( max(sense_score) )
				#if legal merge, merge and break
				if idx_sense == PRE_EDU_SENSE or idx_sense == EDU_SENSE or puncs[idx] in PRE_EDU_PUNCS:
					instance.segments[idx] = reduceds[idx]
					instance.segments_span[idx] = merge_span
					segment_sense_list[idx] = idx_sense
					puncs[idx] = puncs[idx+1]
					del instance.segments[idx+1]
					del instance.segments_span[idx+1]
					del segment_sense_list[idx+1]
					del puncs[idx+1]
					n_segments = len(instance.segments)
					break
	
	def forward_cky_predict(self, instance):
		
		n_segments = len(instance.segments)
		#the first dimension corresponds to the start index
		#the second dimension corresponds to the range(from 0 to n_segments-1)
		cky_table = [None for start_idx in range(n_segments)]
		for start_idx in range(n_segments):
			cky_table[start_idx] =  [ None for cky_range in range(n_segments)]
		print cky_table
		
		#cky table initial condition
		for start_idx in range(n_segments):
			#make a initial candidate
			#the cky_span has only one element and describe itself, it's only for the initialized case
			cky_span_infos = [ CKY_Span_Info(start_idx, 0, 0) ]
			
			#choose the basic initial sense, may be used if cky algorithm refer the sense of a possible child
			if args.args.cky_predict_structure:
				initial_sense = EDU_SENSE
			elif args.args.cky_predict_edu_and_structure:
				initial_sense = PRE_EDU_SENSE
			
			cky_candidate_list = [ CKY_Candidate( cky_span_infos, instance.segments[start_idx], 0, initial_sense, NON_CENTER) ]
			cky_table[start_idx][0] = CKY_Unit(cky_candidate_list)
		
		#cky algorithm
		for cky_range in range(1,n_segments):
			for start_idx in range(0, n_segments-cky_range):
				cky_table[start_idx][cky_range] = CKY_Unit([])
				#each element is [merge_score, cky_candidate] tuple
				cky_candidate_score_list = []
				end_idx = start_idx+cky_range
				#find candidates, middle_index is the end idx of left unit
				#start_idx+cky_range since at least 1 left for the right unit 
				for middle_idx in range(start_idx, start_idx+cky_range):
					cky_unit_left = cky_table[start_idx][middle_idx-start_idx]
					cky_unit_right = cky_table[middle_idx+1][end_idx-middle_idx-1]
					for left_idx in range(len(cky_unit_left.cky_candidate_list)):
						for right_idx in range(len(cky_unit_right.cky_candidate_list)):
							
							left_candidate = cky_unit_left.cky_candidate_list[left_idx]
							right_candidate = cky_unit_right.cky_candidate_list[right_idx]
							left = left_candidate.representation
							right = right_candidate.representation
							
							reduced = self.reduce(left, right)
							label_space = self.struct_linear(reduced)
							label_score = F.log_softmax(label_space)
							merge_score = label_score[0][1].data[0]
							#accumulate the probility scores of left and right
							struct_score = merge_score+left_candidate.score+right_candidate.score
							
							#make the cky_span_info of the now left and right candidates
							cky_span_infos = [ CKY_Span_Info(start_idx, middle_idx-start_idx, left_idx), CKY_Span_Info(middle_idx+1, end_idx-middle_idx-1, right_idx) ]
							#sense and center is temporarily initialized
							cky_candidate = CKY_Candidate(cky_span_infos, reduced, struct_score, NON_SENSE, NON_CENTER)
							cky_candidate_score_list.append([merge_score, cky_candidate])
				
				#sort according to the merge score
				sorted_cky_candidate_score_list = sorted(cky_candidate_score_list, key=lambda x: x[1].score, reverse=True)
				
				candidate_count = 0
				for cky_candidate_score in sorted_cky_candidate_score_list:
					cky_candidate = cky_candidate_score[1]
					infos = cky_candidate.cky_span_infos

					#for center
					label_space = self.center_linear(cky_candidate.representation)
					label_score = F.log_softmax(label_space)
					center_score = [ x.data[0] for x in label_score[0]  ]
					#exclude edu centers
					center_score = center_score[0:3]
					idx_center = center_score.index( max(center_score) )

					#for sense
					label_space = self.sense_linear(cky_candidate.representation)
					label_score = F.log_softmax(label_space)
					sense_score = [ x.data[0] for x in label_score[0]  ]
					
					#for some condition, exclude pre edu senses
					if args.args.cky_predict_structure:
						sense_score = sense_score[0:4]
					elif args.args.cky_predict_edu_and_structure:
						cky_unit_left = cky_table[infos[0].start_idx][infos[0].cky_range]
						cky_unit_right = cky_table[infos[1].start_idx][infos[1].cky_range]		
						left_candidate = cky_unit_left.cky_candidate_list[infos[0].candidate_idx]
						right_candidate = cky_unit_right.cky_candidate_list[infos[1].candidate_idx] 
						left_sense = left_candidate.sense
						right_sense = right_candidate.sense
						# if one of left and right is not pre edu, exclude pre edu senses
						middle_punc = instance.puncs[infos[1].start_idx-1]
						if (left_sense in DU_SENSE or right_sense in DU_SENSE ) or middle_punc in EDU_PUNCS:
							#if middle_punc in EDU_PUNCS:
							sense_score = sense_score[0:4]
					
					idx_sense = sense_score.index( max(sense_score) )	
						
					#force to PRE_EDU_SENSE if the end punc is of certain type
					if args.args.cky_predict_edu_and_structure:
						right_punc = instance.puncs[infos[1].start_idx+infos[1].cky_range]
						if right_punc in PRE_EDU_PUNCS:
							idx_sense = PRE_EDU_SENSE
					
					cky_candidate.sense = idx_sense
					cky_candidate.center = idx_center
					'''
					print 'start_idx:', start_idx, 'cky_range: ', cky_range
					for info in infos:
						print '\t info_start_idx:', info.start_idx, 'info_cky_range: ', info.cky_range, 'sense: ', cky_table[info.start_idx][info.cky_range].cky_candidate_list[info.candidate_idx] .sense
					print 'struct_score: ', cky_candidate_score[1].score, 'merge_score:', cky_candidate_score[0], 'sense: ', cky_candidate.sense
					'''
					if candidate_count == N_CKY_CANDIDATES:
						#break
						continue
					candidate_count += 1
					cky_table[start_idx][cky_range].cky_candidate_list.append(cky_candidate)

		#report.reporter.output_cky_table(cky_table)	
		#empty the i_relations before we fill it with the predicted result 
		instance.i_relations = []
		self.reconstruct_i_relations_from_cky_table(instance, cky_table, cky_table[0][n_segments-1].cky_candidate_list[0])
						
	def forward(self, instance):		
		instance.segments = list( self.double_lstmmodel(instance.segments) )
		#print('initial report:')	
		#report.reporter.show_attributes(instance,object_attr='i_relations',simplify_attr='segments')
		if not args.args.gold_edu_direct:
			if args.args.gold_edu:
				self.forward_process_gold_edu(instance)
			elif args.args.sequence_predict_edu:
				self.forward_sequence_predict_edu(instance)
			elif args.args.greedy_predict_edu:
				self.forward_greedy_predict_edu(instance)
		#print('edu report:')		
		#report.reporter.show_attributes(instance,object_attr='i_relations',simplify_attr='segments')
		#edu_stage
		if args.args.greedy_predict_structure:
			self.forward_greedy_predict_structure(instance)
		elif args.args.cky_predict_structure or args.args.cky_predict_edu_and_structure:
			self.forward_cky_predict(instance)
			#self.forward_greedy_predict_structure(instance)
		print('finish report:')
		report.reporter.show_attributes(instance,object_attr='i_relations',simplify_attr='segments')
		return instance


		
def tree_lstm(c1, c2, lstm_in):
	a, i, f1, f2, o = lstm_in.chunk(5, 1)
	c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
	h = o.sigmoid() * c.tanh()
	return h, c
	
class Reduce(nn.Module):
	def __init__(self, size):
		super(Reduce, self).__init__()
		self.left = nn.Linear(size, 5 * size)
		self.right = nn.Linear(size, 5 * size, bias=False)

	def forward(self, left_in, right_in):
		left = torch.chunk(left_in, 2, 1)
		right = torch.chunk(right_in, 2, 1)
		lstm_in = self.left(left[0]) + self.right(right[0])
		lstm_out = tree_lstm(left[1], right[1], lstm_in)
		return torch.cat(lstm_out, 1)
		
class Double_LSTMModel(nn.Module):
	
	def __init__(self, embedding_dim, word_hidden_dim, seg_hidden_dim, vocab_size, gpu):
		super(Double_LSTMModel, self).__init__()
		self.gpu = gpu
		self.word_hidden_dim = word_hidden_dim
		self.seg_hidden_dim = seg_hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)		
		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.word_lstm = nn.LSTM(embedding_dim, word_hidden_dim)
		self.seg_lstm = nn.LSTM(word_hidden_dim, seg_hidden_dim, bidirectional=True)
		self.word_hidden = self.init_hidden(self.word_hidden_dim)
		self.seg_hidden = self.init_hidden(self.seg_hidden_dim)
		
		
	def init_hidden(self, hidden_dim, bidirectional=False):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		if bidirectional:
			direction = 2
		else:
			direction = 1
		if self.gpu:
			return (autograd.Variable(torch.zeros(direction, 1, hidden_dim)).cuda(),
				autograd.Variable(torch.zeros(direction, 1, hidden_dim)).cuda())
		else:
			return (autograd.Variable(torch.zeros(direction, 1, hidden_dim)),#.cuda(),
				autograd.Variable(torch.zeros(direction, 1, hidden_dim)))#.cuda())
			
		
	def forward(self, sentences):
		self.word_hidden = self.init_hidden(self.word_hidden_dim)
		self.seg_hidden = self.init_hidden(self.seg_hidden_dim, bidirectional=True)
		sent_embeds = []
		for sent in sentences:
			embeds = self.word_embeddings(sent)
			sent_embed = self.word_lstm(embeds.view(len(sent), 1, -1), self.word_hidden)[0]
			sent_embeds.append(sent_embed[-1].view(1, -1))
		#list to tensor, concatenates sequence of tensors along a new dimension(0)
		sent_embeds = torch.stack(sent_embeds)
		sent_embeds = self.seg_lstm(sent_embeds, self.seg_hidden)[0]
		return sent_embeds

