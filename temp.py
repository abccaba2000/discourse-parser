class Instance():
	def __init__(self):
		self.segments = [[...], [...],[...]]
		self.segments_span = [  ]
		self.i_relations = []
		self.label = 'true' or 'false'

class I_Relation():
	def __init__(self, span, sense, center):
		self.span = span 
		self.sense = sense
		self.center = center

class Corpus():
	def __init__(self):
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing
		self.text = ''
		#each element is a [start_idx, end_idx] list, a text is devided to segments by given punctuations
		self.segments_span = []
		#each element is a [start_idx, end_idx] list, some segments merge to a EDU
		self.edus_span = []
		#top-down, post order
		self.relations = []

class Relation():
	def __init__(self, span, sense, center):
		#The span index is start from 0, not 1, different with the xml format
		self.span = span
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing 
		self.sense = sense
		# 1, 2, or 3
		self.center = center
		return
		
		
		
if 1:		
		buffer = list( reversed ( self.lstmmodel(sentences) ) )
		stack = []
		for trans in transitions:
			if trans == 's':
				stack.append(buffer.pop())
			elif trans == 'r':
				right = stack.pop()
				left = stack.pop()
				reduced = self.reduce(left, right)
				stack.append(reduced)
		final_representation = stack.pop()
		if center:
			label_space = self.center_linear(final_representation)
			label_scores = F.log_softmax(label_space)
		elif relation:
			label_space = self.relation_linear(final_representation)
			label_scores = F.log_softmax(label_space)				
		else:
			label_space = self.struct_linear(final_representation)
			label_scores = F.log_softmax(label_space)

		return label_scores
		
no limit		
(0.87355110642781875, 0.81916996047430835, 0.8454869964303926, None)
and
(0.87289915966386555, 0.82114624505928857, 0.8462321792260693, None)
or 
(0.84145176695319956, 0.87055335968379444, 0.85575522098105872, None)
and cky2
(0.86763185108583252, 0.82905138339920947, 0.84790298130368869, None)
or cky2
(0.84291187739463602, 0.86956521739130432, 0.85603112840466933, None)

'./model_join_objective_20epoch' gold edu cky2
gold_n :  579
tp :  374
f1:  0.646499567848
join :
pred_n :  578
gold_n :  579
tp :  203
f1:  0.350907519447
center :
pred_n :  578
gold_n :  579
tp :  223
f1:  0.38547968885
sense :
pred_n :  578
gold_n :  579
tp :  247
f1:  0.426966292135
'./model_join_objective_20epoch' sequence edu cky2
gold_n :  579
tp :  292
f1:  0.513632365875
join :
pred_n :  558
gold_n :  579
tp :  151
f1:  0.265611257696
center :
pred_n :  558
gold_n :  579
tp :  165
f1:  0.290237467018
sense :
pred_n :  558
gold_n :  579
tp :  185
f1:  0.325417766051
(0.89227642276422769, 0.8675889328063241, 0.87975951903807625, None)
'./model_join_objective_20epoch' cky2
gold_n :  579
tp :  301
f1:  0.495065789474
join :
pred_n :  637
gold_n :  579
tp :  163
f1:  0.268092105263
center :
pred_n :  637
gold_n :  579
tp :  175
f1:  0.287828947368
sense :
pred_n :  637
gold_n :  579
tp :  198
f1:  0.325657894737
(0.84925093632958804, 0.89624505928853759, 0.87211538461538463, None)
'./model_20epoch' gold edu cky2
gold_n :  579
tp :  388
f1:  0.67070008643
join :
pred_n :  578
gold_n :  579
tp :  196
f1:  0.338807260156
center :
pred_n :  578
gold_n :  579
tp :  223
f1:  0.38547968885
sense :
pred_n :  578
gold_n :  579
tp :  238
f1:  0.411408815903
'./model_20epoch' sequence edu cky2
gold_n :  579
tp :  287
f1:  0.512957998213
join :
pred_n :  540
gold_n :  579
tp :  128
f1:  0.228775692583
center :
pred_n :  540
gold_n :  579
tp :  149
f1:  0.266309204647
sense :
pred_n :  540
gold_n :  579
tp :  166
f1:  0.296693476318
(0.88008342022940567, 0.83399209486166004, 0.85641806189751402, None)
'./model_20epoch' cky2
gold_n :  579
tp :  297
f1:  0.493765586035
join :
pred_n :  624
gold_n :  579
tp :  141
f1:  0.234413965087
center :
pred_n :  624
gold_n :  579
tp :  163
f1:  0.270989193682
sense :
pred_n :  624
gold_n :  579
tp :  175
f1:  0.290939318371
(0.84291187739463602, 0.86956521739130432, 0.85603112840466933, None)
'./model_right_merge_15epoch' cky2
gold_n :  579
tp :  306
f1:  0.49756097561
join :
pred_n :  651
gold_n :  579
tp :  149
f1:  0.242276422764
center :
pred_n :  651
gold_n :  579
tp :  168
f1:  0.273170731707
sense :
pred_n :  651
gold_n :  579
tp :  185
f1:  0.30081300813
(0.84093023255813959, 0.89328063241106714, 0.86631528509822708, None)
'./model_double_lstm_20_epoch' cky2
pred_n :  676
gold_n :  579
tp :  306
f1:  0.48764940239
join :
pred_n :  676
gold_n :  579
tp :  149
f1:  0.237450199203
center :
pred_n :  676
gold_n :  579
tp :  177
f1:  0.282071713147
sense :
pred_n :  676
gold_n :  579
tp :  179
f1:  0.285258964143
(0.83680870353581138, 0.9120553359683794, 0.87281323877068551, None)

