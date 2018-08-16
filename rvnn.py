# coding=UTF-8
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd

import data_structure


if True:
	EDU_PUNCS = (u'。')
	PRE_EDU_PUNCS = (u'、')
else:
	EDU_PUNCS = ()
	PRE_EDU_PUNCS = ()

N_CKY_CANDIDATES = 2	

greedy_flag = False
gold_edu_flag = False
lstm_flag = True
seg_lstm_flag = False
left_seq_flag = True
right_seq_flag = False
GPU = True


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
		
class RVNN(nn.Module):
	def __init__(self, config):
		super(RVNN, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.d_embedding)
		self.reduce = Reduce(config.d_rvhidden/2)
		self.struct_linear = nn.Linear(config.d_rvhidden, config.d_struct)
		self.center_linear = nn.Linear(config.d_rvhidden, config.d_center)
		self.sense_linear = nn.Linear(config.d_rvhidden, config.d_relation)
		#whether to use lstm to process segments 
		if lstm_flag:
			self.d_lstm_hidden = config.d_rvhidden
			self.lstm = nn.LSTM(config.d_embedding, self.d_lstm_hidden)
			self.gpu = config.gpu
			if seg_lstm_flag:
				self.d_seg_lstm_hidden = self.d_lstm_hidden/2
				self.seg_lstm = nn.LSTM(self.d_lstm_hidden, self.d_seg_lstm_hidden, bidirectional=True)
	def init_hidden(self, hidden_dim, bidirectional=False):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		if bidirectional:
			direction = 2
		else:
			direction = 1
		#if self.gpu:
		if GPU:
			return (autograd.Variable(torch.zeros(direction, 1, hidden_dim)).cuda(),
				autograd.Variable(torch.zeros(direction, 1, hidden_dim)).cuda())
		else:
			return (autograd.Variable(torch.zeros(direction, 1, hidden_dim)),#.cuda(),
				autograd.Variable(torch.zeros(direction, 1, hidden_dim)))#.cuda())
			
	def reconstruct_i_relations_from_cky_table(self, instance, cky_table, cky_candidate):
	
		infos = cky_candidate.cky_span_infos		
		#recursive to the bottom
		if len(infos) == 1:
			return instance
		
		left_start_idx = infos[0].start_idx
		left_end_idx = infos[0].start_idx+infos[0].cky_range
		left_span = [ instance.fragment_spans[left_start_idx][0], instance.fragment_spans[left_end_idx][-1] ]
		
		right_start_idx = infos[-1].start_idx
		right_end_idx = infos[-1].start_idx+infos[-1].cky_range
		right_span = [ instance.fragment_spans[right_start_idx][0], instance.fragment_spans[right_end_idx][-1] ]
		
		i_relation = data_structure.I_Relation([left_span, right_span], cky_candidate.sense, cky_candidate.center, '')
		
		#modify the instance directly
		instance.i_relations.append(i_relation)
		
		#find next recursive target
		for info in infos:
			next_candidate = cky_table[info.start_idx][info.cky_range].cky_candidate_list[info.candidate_idx]
			self.reconstruct_i_relations_from_cky_table(instance, cky_table, next_candidate)
		return instance
		
	def forward_by_i_relations(self, instance, struct=False, center=False, sense=False):
		#convert segments to list of word embeddings
		if lstm_flag:
			self.lstm_hidden = self.init_hidden(self.d_lstm_hidden)
			fragments = []
			#used to modify the index of spans to fit instance.fragments
			start_idx = instance.segment_spans[0][0]
			for span in instance.segment_spans:
				#print instance.fragments
				#print span
				fragment = instance.fragments[span[0]-start_idx:span[-1]+1-start_idx]
				#print fragment
				fragment = self.word_embeddings(fragment)
				#print fragment
				fragment = self.lstm(fragment.view(span[-1]-span[0]+1, 1, -1), self.lstm_hidden)[0]
				fragments.append(fragment[-1].view(1, -1))
			if seg_lstm_flag:
				self.seg_hidden = self.init_hidden(self.d_seg_lstm_hidden, bidirectional=True)
				seg_embeds = torch.stack(fragments)
				fragments = list(self.seg_lstm(seg_embeds, self.seg_hidden)[0])				
			instance.fragments = fragments
			instance.fragment_spans = instance.segment_spans
		else:
			instance.fragments = list(self.word_embeddings(instance.fragments))
		#print instance.fragments
		for i_relation in  instance.i_relations:
			left_span = i_relation.spans[0]
			for idx in range(len(instance.fragment_spans)):
				#since the i_relations is in post order(buttom up stlye), it's always to find a corresponding fragment to merge
				if left_span == instance.fragment_spans[idx]:
					#data_structure.print_train_instance(instance)
					#data_structure.print_i_relation(i_relation)
					left = instance.fragments[idx].view(1, -1)
					right = instance.fragments[idx+1].view(1, -1)
					reduced = self.reduce(left, right)
					instance.fragments[idx] = reduced
					del instance.fragments[idx+1]
					#modify the segment span list to follow the change of segment list
					instance.fragment_spans[idx] = [ instance.fragment_spans[idx][0], instance.fragment_spans[idx+1][-1] ]
					del instance.fragment_spans[idx+1]
					#finish the merge, break to deal with next relation
					break
		final_representation = instance.fragments[0]
		
		scores_list = []
		
		if struct:
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
				

		
		return instance, scores_list

	def forward_process_gold_edu(self, instance):
		#pre_edustage
		n_fragments = len(instance.fragments)
		#use flag instead of use torch Variable itself since the comparison limit, ex: can't write "if (torch Variable) == 0:" 
		left_flag = False
		edus = []
		edu_idx = 0
		left = None 
		#from left to right
		for idx in range(n_fragments):
			edu_span = instance.edu_spans[edu_idx]
			# start a new edu or merge to previous edu
			if left_flag == False:
				left = instance.fragments[idx]
				left_span = instance.fragment_spans[idx]
			else:
				right = instance.fragments[idx]
				right_span = instance.fragment_spans[idx]
				left = self.reduce(left, right)
				merge_span = [left_span, right_span]
				left_span = [left_span[0], right_span[1]]	
			#if reaching the edu boundary
			if left_span[-1] == edu_span[-1]:
				edus.append(left)
				if left_flag:
					i_relation = data_structure.I_Relation(merge_span,data_structure.SENSE_TO_LABEL[data_structure.EDU_SENSE] , data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER], '')
					instance.i_relations.append(i_relation)
				left = None
				left_flag = False
				edu_idx += 1
			else:
				if left_flag:
					i_relation = data_structure.I_Relation(merge_span,data_structure.SENSE_TO_LABEL[data_structure.PRE_EDU_SENSE] , data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER], '')
					instance.i_relations.append(i_relation)
				left_flag = True
		# set the edus, modify the instance segments to edu level
		instance.fragments = edus
		instance.fragment_spans = instance.edu_spans
		return instance
		
	def forward_sequence_predict_edu(self, instance):
	
		#pre_edustage
		n_fragments = len(instance.fragments)
		#use flag instead of use torch Variable itself since the comparison limit, ex: can't write "if (torch Variable) == 0:" 
		left_flag = False
		right_flag = False
		edus = []
		edus_span = []
		#from left to right
		for i in range(n_fragments):
			#the code is written for left-to-right logic, when we need to process right to left, just modify the idx, other variables(inculding left, right) remain the same
			if left_seq_flag:
				idx = i
			else:
				idx = n_fragments-i-1
			punc = instance.puncs[idx]
			#merge or just set the left segment
			if not left_flag:
				left = instance.fragments[idx]
				left_span = instance.fragment_spans[idx]
				left_flag = True
			else:
				#merge, make sense
				right = instance.fragments[idx]
				right_span = instance.fragment_spans[idx]
				right_flag = True
				reduced = self.reduce(left, right)
				merge_span = [left_span[0], right_span[1]]	
				label_space = self.sense_linear(reduced)
				label_score = F.log_softmax(label_space)
				sense_score = [ x.data[0] for x in label_score[0]  ]
				merge_sense = sense_score.index( max(sense_score) )
			
			#modify the instance information, no need to deal with the i_reltion
			if right_flag:
				if merge_sense == data_structure.SENSE_TO_LABEL[data_structure.EDU_SENSE] and punc not in PRE_EDU_PUNCS:
					edus.append(reduced)
					edus_span.append(merge_span)
					merge_sense = data_structure.SENSE_TO_LABEL[data_structure.EDU_SENSE]
					i_relation = data_structure.I_Relation([left_span, right_span], merge_sense, data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER], '')
					instance.i_relations.append(i_relation)
					left_flag = False
				else:
					if merge_sense == data_structure.SENSE_TO_LABEL[data_structure.PRE_EDU_SENSE] or punc in PRE_EDU_PUNCS:
						merge_sense = data_structure.SENSE_TO_LABEL[data_structure.PRE_EDU_SENSE]
						i_relation = data_structure.I_Relation([left_span, right_span], merge_sense, data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER], '')
						instance.i_relations.append(i_relation)
						left = reduced
						left_span = merge_span

						#condition that the sense is in du sense
					else:
						edus.append(left)
						edus_span.append(left_span)
						left = right
						left_span = right_span
					#left is already modified above
					if idx == n_fragments-1 or punc in EDU_PUNCS:
						edus.append(left)
						edus_span.append(left_span)
						left_flag = False
				
			else:
				if idx == n_fragments-1 or punc in EDU_PUNCS:
					edus.append(left)
					edus_span.append(left_span)
					left_flag = False						
				
			right_flag = False
			
		# set the edus, modify the instance segments to edu level
		instance.fragments = edus
		instance.fragment_spans = edus_span
		return instance

	def forward_greedy_predict_structure(self, instance):
		i_relations = []
		while len(instance.fragments) > 1:
			n_fragments = len(instance.fragments)
			scores = []
			reduceds = []
			for i in range(n_fragments-1):
				reduced = self.reduce(instance.fragments[i], instance.fragments[i+1])
				reduceds.append(reduced)
				label_space = self.struct_linear(reduced)
				label_score = F.log_softmax(label_space)
				scores.append(label_score[0][1].data[0])
			#for span
			idx = scores.index( max(scores) )
			left_span = instance.fragment_spans[idx]
			right_span = instance.fragment_spans[idx+1]
			i_relation_span = [left_span, right_span]
			#print 'left_span: ', left_span, 'right_span: ', right_span, 'max score: ', max(scores)
			#for center
			label_space = self.center_linear(reduceds[idx])
			label_score = F.log_softmax(label_space)
			center_score = [ x.data[0] for x in label_score[0]  ]
			#exclude non du centers
			lowest = min(center_score)
			for i in range(len(center_score)):
				if i not in data_structure.DU_CENTER_LABEL:
					center_score[i] = lowest-1
			idx_center = center_score.index( max(center_score) )

				
			
			#for sense
			label_space = self.sense_linear(reduceds[idx])
			label_score = F.log_softmax(label_space)
			sense_score = [ x.data[0] for x in label_score[0]  ]
			#exclude non du senses
			lowest = min(sense_score)
			for i in range(len(sense_score)):
				if i not in data_structure.DU_SENSE_LABEL:
						sense_score[i] = lowest-1
			idx_sense = sense_score.index( max(sense_score) )
				
			#modify the instance information
			instance.fragments[idx] = reduceds[idx]
			instance.fragment_spans[idx] = [ i_relation_span[0][0], i_relation_span[-1][-1] ]
			del instance.fragments[idx+1]
			del instance.fragment_spans[idx+1]
			#make a new i_relation
			i_relation = data_structure.I_Relation(i_relation_span, idx_sense, idx_center, '')
			i_relations.append(i_relation)
		instance.i_relations = i_relations
		return instance
		
	def forward_cky_predict(self, instance, word_level_flag=True, du_level_flag=False):
		
		n_fragments = len(instance.fragments)
		#the first dimension corresponds to the start index
		#the second dimension corresponds to the range(from 0 to n_fragments-1)
		cky_table = [None for start_idx in range(n_fragments)]
		for start_idx in range(n_fragments):
			cky_table[start_idx] =  [ None for cky_range in range(n_fragments)]
		
		#cky table initial condition
		for start_idx in range(n_fragments):
			#make a initial candidate
			#the cky_span has only one element and describe itself, it's only for the initialized case
			cky_span_infos = [ CKY_Span_Info(start_idx, 0, 0) ]
			
			#choose the basic initial sense, may be used if cky algorithm refer the sense of a possible child
			'''
			if args.args.cky_predict_structure:
				initial_sense = data_structure.SENSE_TO_LABEL[EDU_SENSE]
			elif args.args.cky_predict_edu_and_structure:
				initial_sense = data_structure.SENSE_TO_LABEL[PRE_EDU_SENSE]
			'''
			initial_sense = data_structure.SENSE_TO_LABEL[data_structure.PRE_EDU_SENSE]
			
			#view(1,-1) for Reduce function
			cky_candidate_list = [ CKY_Candidate( cky_span_infos, instance.fragments[start_idx].view(1,-1), 0, initial_sense, data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER]) ]
			cky_table[start_idx][0] = CKY_Unit(cky_candidate_list)
		
		#cky algorithm
		for cky_range in range(1,n_fragments):
			for start_idx in range(0, n_fragments-cky_range):
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
							#print 'label_score', label_score
							merge_score = label_score[0][1].data[0]
							#accumulate the probility scores of left and right
							struct_score = merge_score+left_candidate.score+right_candidate.score
							
							#make the cky_span_info of the now left and right candidates
							cky_span_infos = [ CKY_Span_Info(start_idx, middle_idx-start_idx, left_idx), CKY_Span_Info(middle_idx+1, end_idx-middle_idx-1, right_idx) ]
							#sense and center is temporarily initialized
							cky_candidate = CKY_Candidate(cky_span_infos, reduced, struct_score, data_structure.SENSE_TO_LABEL[data_structure.PSEUDO_SENSE], data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER])
							cky_candidate_score_list.append([merge_score, cky_candidate])
				
				#sort according to the merge score
				sorted_cky_candidate_score_list = sorted(cky_candidate_score_list, key=lambda x: x[1].score, reverse=True)
				
				candidate_count = 0
				for cky_candidate_score in sorted_cky_candidate_score_list:
					cky_candidate = cky_candidate_score[1]
					infos = cky_candidate.cky_span_infos
					if word_level_flag:
						cky_candidate.sense = data_structure.SENSE_TO_LABEL[data_structure.PSEUDO_SENSE]
						cky_candidate.center = data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER]
					else:
						#for center
						label_space = self.center_linear(cky_candidate.representation)
						label_score = F.log_softmax(label_space)
						center_score = [ x.data[0] for x in label_score[0]  ]
						#exclude edu centers
						lowest = min(center_score)
						for idx in range(len(center_score)):
							if idx not in data_structure.DU_CENTER_LABEL:
								center_score[idx] = lowest-1
						idx_center = center_score.index( max(center_score) )

						#for sense
						label_space = self.sense_linear(cky_candidate.representation)
						label_score = F.log_softmax(label_space)
						sense_score = [ x.data[0] for x in label_score[0]  ]
						
						#for some condition, exclude pre edu senses
						'''
						if args.args.cky_predict_structure:
							sense_score = sense_score[0:4]
						elif args.args.cky_predict_edu_and_structure:
						'''
						cky_unit_left = cky_table[infos[0].start_idx][infos[0].cky_range]
						cky_unit_right = cky_table[infos[1].start_idx][infos[1].cky_range]		
						left_candidate = cky_unit_left.cky_candidate_list[infos[0].candidate_idx]
						right_candidate = cky_unit_right.cky_candidate_list[infos[1].candidate_idx] 
						left_sense = left_candidate.sense
						right_sense = right_candidate.sense
						# if one of left and right is not pre edu, exclude pre edu senses
						# or du_level_flag is true
						middle_punc = instance.puncs[infos[1].start_idx-1]
						right_punc = instance.puncs[infos[1].start_idx+infos[1].cky_range]						
						if (left_sense in data_structure.DU_SENSE_LABEL or right_sense in data_structure.DU_SENSE_LABEL ) or middle_punc in EDU_PUNCS or du_level_flag:
							lowest = min(sense_score)
							for idx in range(len(sense_score)):
								if idx not in data_structure.DU_SENSE_LABEL or\
								idx == data_structure.SENSE_TO_LABEL[data_structure.EDU_SENSE]:
									sense_score[idx] = lowest-1
							idx_sense = sense_score.index( max(sense_score) )	
							
						#force to PRE_EDU_SENSE if the end punc is of certain type					
						elif right_punc in PRE_EDU_PUNCS:
							idx_sense = data_structure.SENSE_TO_LABEL[data_structure.PRE_EDU_SENSE]
						else:
							idx_sense = sense_score.index( max(sense_score) )
						
						if idx_sense == data_structure.SENSE_TO_LABEL[data_structure.EDU_SENSE]:
							idx_center = data_structure.CENTER_TO_LABEL[data_structure.NON_CENTER]

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
		#the final representation replace the original instance.fragments
		instance.fragments = [cky_table[0][-1].cky_candidate_list[0].representation]
		instance = self.reconstruct_i_relations_from_cky_table(instance, cky_table, cky_table[0][n_fragments-1].cky_candidate_list[0])
		return instance
						
	def forward(self, instance):		
		#list of list
		fragments = []
		w_i_relations_list = []
		
		if lstm_flag:
			for span in instance.segment_spans:
				fragment = instance.fragments[span[0]:span[-1]+1]
				#print fragment
				fragment = self.word_embeddings(fragment)
				#print fragment
				fragment = self.lstm(fragment.view(span[-1]-span[0]+1, 1, -1), self.lstm_hidden)[0]
				fragments.append(fragment[-1].view(1, -1))
				print instance.id
			if seg_lstm_flag:
				self.seg_hidden = self.init_hidden(self.d_seg_lstm_hidden, bidirectional=True)
				seg_embeds = torch.stack(fragments)
				fragments = list(self.seg_lstm(seg_embeds, self.seg_hidden)[0])
		else:
			#we construct each segment parsing tree first		
			for seg_span in instance.segment_spans:
				seg_instance = data_structure.Instance()
				seg_instance.fragments = instance.fragments[seg_span[0]:seg_span[-1]+1]
				seg_instance.fragment_spans = [[i,i] for i in range(seg_span[0], seg_span[-1]+1)]
				#convert segments to list of word embeddings
				seg_instance.fragments = self.word_embeddings(seg_instance.fragments)
				seg_instance = self.forward_cky_predict(seg_instance, word_level_flag=True)
				#get the word level i_relations of the segment
				w_i_relations_list.append(seg_instance.i_relations)
				#get the final representation of the segment
				fragments.append(seg_instance.fragments[0])
		
		instance.w_i_relations_list = w_i_relations_list
		#construct the discourse parsing tree based on segments
		instance.fragments = fragments
		instance.fragment_spans = instance.segment_spans
		#empty the i_relations before we fill it with the predicted result 
		instance.i_relations = []
		#data_structure.print_test_instance(instance)
		du_level_flag = False
		if gold_edu_flag:
			instance = self.forward_process_gold_edu(instance)
			du_level_flag = True
		elif left_seq_flag:
			#will append the resulting i_relations in instance.i_relations
			instance = self.forward_sequence_predict_edu(instance)
			du_level_flag = True
		#will append the resulting i_relations in instance.i_relations
		#data_structure.print_test_instance(instance)
		if greedy_flag:
			instance = self.forward_greedy_predict_structure(instance)
		else:
			instance = self.forward_cky_predict(instance, word_level_flag=False, du_level_flag=du_level_flag)
		#data_structure.print_test_instance(instance)
		#empty the relation before predicting
		instance.du_i_relations = []
		instance.seg_i_relations = []
		for i_relation in instance.i_relations:
			if i_relation.sense in data_structure.DU_SENSE_LABEL and\
			i_relation.sense != data_structure.SENSE_TO_LABEL[data_structure.EDU_SENSE]:
				instance.du_i_relations.append(i_relation)
			else:
				instance.seg_i_relations.append(i_relation)
		
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

class BiLSTM_CRF(nn.Module):
	def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
		super(BiLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		#+2 for start and stop
		self.tagset_size = tagset_size+2

		self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
							num_layers=1, bidirectional=True)

		# Maps the output of the LSTM into tag space.
		self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

		# Matrix of transition parameters.  Entry i,j is the score of
		# transitioning *to* i *from* j.
		self.transitions = nn.Parameter(
			torch.randn(self.tagset_size, self.tagset_size))

		self.START_TAG_LABEL = tagset_size
		self.STOP_TAG_LABEL = tagset_size+1

		# These two statements enforce the constraint that we never transfer
		# to the start tag and we never transfer from the stop tag
		self.transitions.data[self.START_TAG_LABEL, :] = -10000
		self.transitions.data[:, self.STOP_TAG_LABEL] = -10000
		
		self.hidden = self.init_hidden()
		
		

	def init_hidden(self):
		#return (torch.randn(2, 1, self.hidden_dim // 2),
		#		torch.randn(2, 1, self.hidden_dim // 2))
		if GPU:
			return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)).cuda(),
				autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)).cuda())
		else:
			return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)),
				autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)))
				
				
	def _forward_alg(self, feats):
		# Do the forward algorithm to compute the partition function
		#init_alphas = [-10000.]*self.tagset_size
		#init_alphas = torch.Tensor(init_alphas).view(1, -1)
		init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
		
		# START_TAG has all of the score.
		init_alphas[0][self.START_TAG_LABEL] = 0.
		
		
		# Wrap in a variable so that we will get automatic backprop
		#forward_var = init_alphas
		if GPU:
			forward_var = autograd.Variable(init_alphas).cuda()
		else:
			forward_var = autograd.Variable(init_alphas)
		
		# Iterate through the sentence
		for feat in feats:
			alphas_t = []  # The forward tensors at this timestep
			for next_tag in range(self.tagset_size):
				# broadcast the emission score: it is the same regardless of
				# the previous tag
				emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
				# the ith entry of trans_score is the score of transitioning to
				# next_tag from i
				trans_score = self.transitions[next_tag].view(1, -1)
				# The ith entry of next_tag_var is the value for the
				# edge (i -> next_tag) before we do log-sum-exp
				
				next_tag_var = forward_var + trans_score + emit_score
				# The forward variable for this tag is log-sum-exp of all the
				# scores.
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			forward_var = torch.cat(alphas_t).view(1, -1)
		terminal_var = forward_var + self.transitions[self.STOP_TAG_LABEL]
		alpha = log_sum_exp(terminal_var)

		return alpha

	def _get_lstm_features(self, sentence):
		self.hidden = self.init_hidden()
		embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
		lstm_out, self.hidden = self.lstm(embeds, self.hidden)
		lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
		lstm_feats = self.hidden2tag(lstm_out)
		return lstm_feats

	def _score_sentence(self, feats, tags):
		# Gives the score of a provided tag sequence
		#score = torch.zeros(1)
		#tags = torch.cat([torch.tensor([self.START_TAG_LABEL], dtype=torch.long), tags])
		
		if GPU:
			score = autograd.Variable(torch.Tensor([0])).cuda()
			tags = torch.cat([torch.LongTensor([self.START_TAG_LABEL]).cuda(), tags])
		else:
			score = autograd.Variable(torch.Tensor([0]))
			tags = torch.cat([torch.LongTensor([self.START_TAG_LABEL]), tags])
		
		#tags = torch.cat([torch.LongTensor([self.START_TAG_LABEL]), tags])
		
		for i, feat in enumerate(feats):
			score = score + \
				self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
		score = score + self.transitions[self.STOP_TAG_LABEL, tags[-1]]
		return score

	def _viterbi_decode(self, feats):
		backpointers = []

		# Initialize the viterbi variables in log space
		#init_vvars = [-10000.]*self.tagset_size
		#init_vvars = torch.Tensor(init_vvars).view(1, -1)		
		#init_vvars = torch.full((1, self.tagset_size), -10000.)
		init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
		init_vvars[0][self.START_TAG_LABEL] = 0

		# forward_var at step i holds the viterbi variables for step i-1
		if GPU:
			forward_var = autograd.Variable(init_vvars).cuda()
		else:
			forward_var = autograd.Variable(init_vvars)
		
		
		for feat in feats:
			bptrs_t = []  # holds the backpointers for this step
			viterbivars_t = []  # holds the viterbi variables for this step

			for next_tag in range(self.tagset_size):
				# next_tag_var[i] holds the viterbi variable for tag i at the
				# previous step, plus the score of transitioning
				# from tag i to next_tag.
				# We don't include the emission scores here because the max
				# does not depend on them (we add them in below)
				next_tag_var = forward_var + self.transitions[next_tag]
				best_tag_id = argmax(next_tag_var)
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			# Now add in the emission scores, and assign forward_var to the set
			# of viterbi variables we just computed
			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)

		# Transition to STOP_TAG
		terminal_var = forward_var + self.transitions[self.STOP_TAG_LABEL]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		# Follow the back pointers to decode the best path.
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		# Pop off the start tag (we dont want to return that to the caller)
		start = best_path.pop()
		assert start == self.START_TAG_LABEL  # Sanity check
		best_path.reverse()
		return path_score, best_path

	def neg_log_likelihood(self, sentence, tags):
		feats = self._get_lstm_features(sentence)	 
		forward_score = self._forward_alg(feats)
		gold_score = self._score_sentence(feats, tags)
		return forward_score - gold_score
		#return forward_score
		
	def forward(self, sentence):  # dont confuse this with _forward_alg above.
		# Get the emission scores from the BiLSTM
		lstm_feats = self._get_lstm_features(sentence)

		# Find the best path, given the features.
		score, tag_seq = self._viterbi_decode(lstm_feats)
		return score, tag_seq	

def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	#return idx.item()
	return to_scalar(idx)
	
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + \
		torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))		

def to_scalar(var): #var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]