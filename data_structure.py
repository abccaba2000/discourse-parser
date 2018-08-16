# coding=UTF-8
# for parsing xml file
import xml.etree.ElementTree as ET
# for dealing with file system
import os
#for argv
import sys
# for save/load dictionary
import pickle
# for copy objects
import copy
from collections import defaultdict
import random

#whether to only consider segment and discourse unit level relations only when training
seg_du_flag = True

#need special judge for '——', '。」' condition
_ENDs = (u'?', u'”', u'…', u'—', u'、', u'。', u'」', u'！', u'，', u'：', u'；', u'？')
#word_to_index dictionary 
word_to_ix = {}
#for out of vocabulary dictionary index
oov = 0
#file path the dictionary dump
WORD_TO_IX_DUMP_PATH = './word_to_ix'
#center value from xml format to Corpus format
XML_TO_CORPUS_CERTER_DICT = {'1':'former', '2':'latter', '3':'equal'}

PRE_EDU_SENSE = 'PRE_EDU_SENSE'
EDU_SENSE = 'EDU_SENSE'
NON_CENTER = 'NON_CENTER'
PSEUDO_SENSE = 'PSEUDO_SENSE'
PSEUDO_CENTER = 'PSEUDO_CENTER'
NON_TYPE = u''

LABEL_TO_SENSE = {0:'Coordination', 1:'Causality', 2:'Transition', 3:'Explanation'}
LABEL_TO_CENTER = {0:'Front', 1:'Latter', 2:'Equal'}

SENSE_TO_LABEL = {u'并列关系': 0, u'顺承关系': 0, u'递进关系':0, u'选择关系': 0, u'对比关系':0, u'因果关系': 1, u'推断关系': 1, u'假设关系': 1, u'目的关系': 1, u'条件关系': 1, u'背景关系': 1, u'转折关系': 2, u'让步关系': 2, u'解说关系': 3, u'总分关系': 3, u'例证关系': 3, u'评价关系': 3, EDU_SENSE: 4, PRE_EDU_SENSE: 5, PSEUDO_SENSE: 6}
SENSE_DIM = len(set(SENSE_TO_LABEL.values()))
DU_SENSE_LABEL = tuple( SENSE_TO_LABEL[x] for x in (u'并列关系', u'顺承关系', u'递进关系', u'选择关系', u'对比关系', u'因果关系', u'推断关系', u'假设关系', u'目的关系', u'条件关系', u'背景关系', u'转折关系', u'让步关系', u'解说关系', u'总分关系', u'例证关系', u'评价关系', EDU_SENSE) )


CENTER_TO_LABEL = {'former':0, 'latter':1, 'equal':2, NON_CENTER: 3, PSEUDO_CENTER: 3}
CENTER_DIM = len(set(CENTER_TO_LABEL.values()))
DU_CENTER_LABEL = tuple(CENTER_TO_LABEL[x] for x in ('former', 'latter', 'equal')) 
COORD_SENSE_LABEL = SENSE_TO_LABEL[u'并列关系']
COORD_CENTER_LABEL = CENTER_TO_LABEL['equal']


#for lstm-crf 
#extract from corpus, lack ON, IJ  when comparing to pos tag guideline, and NR-SHORT, NN-SHORT, NT-SHORT does not appear in the guideline
POS_TO_LABEL34 = {u'SP': 0, u'BA': 1, u'FW': 2, u'DER': 3, u'DEV': 4, u'MSP': 5, u'ETC': 6, u'JJ': 7, u'DT': 8, u'DEC': 9, u'DEG': 10, u'LB': 11, u'LC': 12, u'NN': 13, u'PU': 14, u'NR': 15, u'PN': 16, u'VA': 17, u'VC': 18, u'AD': 19, u'CC': 20, u'VE': 21, u'M': 22, u'CD': 23, u'P': 24, u'AS': 25, u'NR-SHORT': 26, u'VV': 27, u'CS': 28, u'NT': 29, u'OD': 30, u'NN-SHORT': 31, u'SB': 32, u'NT-SHORT': 33}

#V:0, N:1, LC:2, PN:3, D:4, M:5, AD:6, PP:7, C:8, P:9, O:10
POS_TO_LABEL11 = {u'SP': 9, u'BA': 10, u'FW': 10, u'DER': 9, u'DEV': 9, u'MSP': 9, u'ETC': 9, u'JJ': 10, u'DT': 4, u'DEC': 9, u'DEG': 9, u'LB': 10, u'LC': 2, u'NN': 1, u'PU': 10, u'NR': 1, u'PN': 3, u'VA': 0, u'VC': 0, u'AD': 6, u'CC': 8, u'VE': 0, u'M': 5, u'CD': 4, u'P': 7, u'AS': 9, u'NR-SHORT': 1, u'VV': 0, u'CS': 8, u'NT': 1, u'OD': 4, u'NN-SHORT': 1, u'SB': 10, u'NT-SHORT': 1}

#Verb:0, Noun:1, Conjunction:2, Punctuation:3, other:4
POS_TO_LABEL5 = {u'SP': 4, u'BA': 4, u'FW': 4, u'DER': 4, u'DEV': 4, u'MSP': 4, u'ETC': 4, u'JJ': 4, u'DT': 4, u'DEC': 4, u'DEG': 4, u'LB': 4, u'LC': 4, u'NN': 1, u'PU': 3, u'NR': 1, u'PN': 1, u'VA': 0, u'VC': 0, u'AD': 4, u'CC': 2, u'VE': 0, u'M': 4, u'CD': 4, u'P': 4, u'AS': 4, u'NR-SHORT': 1, u'VV': 0, u'CS': 2, u'NT': 1, u'OD': 4, u'NN-SHORT': 1, u'SB': 4, u'NT-SHORT': 1}

POS_TO_LABEL1 = {u'SP': 0, u'BA': 0, u'FW': 0, u'DER': 0, u'DEV': 0, u'MSP': 0, u'ETC': 0, u'JJ': 0, u'DT': 0, u'DEC': 0, u'DEG': 0, u'LB': 0, u'LC': 0, u'NN': 0, u'PU': 0, u'NR': 0, u'PN': 0, u'VA': 0, u'VC': 0, u'AD': 0, u'CC': 0, u'VE': 0, u'M': 0, u'CD': 0, u'P': 0, u'AS': 0, u'NR-SHORT': 0, u'VV': 0, u'CS': 0, u'NT': 0, u'OD': 0, u'NN-SHORT': 0, u'SB': 0, u'NT-SHORT': 0}

WORD_SIMPLE = u'S'
WORD_BEGIN = u'B'
WORD_MIDDLE = u'M'
WORD_END = u'E'

WORD_TAG_TO_LABEL = {WORD_SIMPLE:0, WORD_BEGIN:1, WORD_MIDDLE:2, WORD_END:3}

POS_TO_LABEL = POS_TO_LABEL1

pos_label_n = max( POS_TO_LABEL.values() )+1
word_tag_label_n = max( WORD_TAG_TO_LABEL.values() )+1
SEQ_TAG_TO_LABEL = {}
#each element is a tuple (word_tag, pos) in which pos is one of pos tag of the label value
#use defaultdict for the condition that the lstm-crf model returns a START or STOP tag
LABEL_TO_WORD_INF = defaultdict(tuple)
for word_tagk, word_tagv in WORD_TAG_TO_LABEL.iteritems():
	for posk, posv in POS_TO_LABEL.iteritems():
		label = word_tagv*pos_label_n+posv
		SEQ_TAG_TO_LABEL[word_tagk+u'-'+posk] = label
		LABEL_TO_WORD_INF[label] = (word_tagk, posk)

#print SEQ_TAG_TO_LABEL
#print LABEL_TO_WORD_INF

STRUCT_LABEL_TRUE = 1
STRUCT_LABEL_FALSE = 0
STRUCT_LABEL_DIM = 2


		

'''
define the Corpus class which contains information of a paragraph: text, segments index, relations and the merge structure

merge structure: words merge to segments, segments merge to EDU, EDUS merge to bigger discourse unit 

'''
class Corpus():
	def __init__(self):
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing
		self.text = ''
		#each element is a Word object, some characters merge to words 
		self.words = []
		#each element is a [start_idx, end_idx] list, a text is devided to segments by given punctuations
		self.segment_spans = []
		#each element is a [start_idx, end_idx] list, some segments merge to a EDU
		self.edu_spans = []
		#relations of discouse unit level, pre order
		self.du_relations = []
		#relations of segment level, pre order
		self.seg_relations = []
		#relations of character level, pre order
		self.w_relations = []
		#filename+'-'+paragraph_count ex: 001.xml-1, count from 1
		self.id = ''
		#check whether the edu span and segment span are coodinated
	def span_certification(self):
		seg_idx = 0
		passed = True
		ends = []
		for e_span in self.edu_spans:
			start = e_span[0]
			end = e_span[-1]
			while start != self.segment_spans[seg_idx][0]:
				ends.append(self.segment_spans[seg_idx][-1])
				seg_idx += 1
				if seg_idx == len(self.segment_spans):
					print 'certification not passed(start):', self.id, start, self.edu_spans
					passed = False
					break
			
			while end != self.segment_spans[seg_idx][-1]:
				ends.append(self.segment_spans[seg_idx][-1])
				seg_idx += 1

				if seg_idx == len(self.segment_spans):
					print 'certification not passed(end):', self.id, end, self.edu_spans
					print ends
					passed = False
					break
			if not passed:
				break
		return

class Word():
	def __init__(self, span, pos):
		self.span = span
		self.pos = pos

def word_to_labels(word):
	char_n = word.span[-1] - word.span[0] + 1
	labels = []
	word_tags = []
	if char_n == 1:
		word_tags.append(WORD_SIMPLE)
	else:
		word_tags.append(WORD_BEGIN)
		for i in range(char_n-2):
			word_tags.append(WORD_MIDDLE)
		word_tags.append(WORD_END)
	for word_tag in word_tags:
		key = word_tag+u'-'+word.pos
		labels.append(SEQ_TAG_TO_LABEL[key])
	return labels
		
		
def labels_to_words_in_test_instance(labels, instance):
	words = []
	start_idx = instance.segment_spans[0][0]
	end_idx = instance.segment_spans[-1][-1]
	char_idx = start_idx
	word_start_idx = char_idx
	for label in labels:
		tag_tuple = LABEL_TO_WORD_INF[label]
		word_tag = tag_tuple[0]
		pos = tag_tuple[1]
		if word_tag == WORD_BEGIN:
			word_start_idx = char_idx
		elif word_tag == WORD_MIDDLE:
			pass
		elif word_tag == WORD_END:
			span = [word_start_idx, char_idx]
			word = Word(span, pos)
			words.append(word)
		elif word_tag == WORD_SIMPLE:
			span = [char_idx, char_idx]
			words.append(word)
		else:
			print 'word tag exception'
			print_test_instance(instance)
		char_idx += 1
	
	instance.words = words 
		
		
class Relation():
	def __init__(self, spans, sense, center, type):
		#The span index is start from 0, not 1, different with the xml format, ex[[0,5],[6,10],[11,13]]
		self.spans = spans
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing 
		self.sense = sense
		#unicode, 'former', 'latter', 'equal', different from xml format
		self.center = center
		#connective type, utf-8, explicit or implicit
		self.type = type
		return		

'''use to store the syntactic parsing information of a segment'''
class ParseSeg():
	def __init__(self, text, relations, words, id):
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing
		self.text = text		
		#pre order
		self.relations = relations 		
		#filename+'-'+segment_count ex: chtb_0001.nw.new-1, count from 1
		self.id = id		
		#each element is a Word object, some characters merge to words
		self.words = words
		
#instance for nn model
class Instance(object):
	def __init__(self):
		#words initially, merging with each each other during the construction process 
		self.fragments = []
		#word spans initially, merging with each other during the construction process 
		self.fragment_spans = []	
		#post order
		self.i_relations = []
		#each element is a Word object list for a segment, some characters merge to words
		self.words_list = []
	
#training instance for nn model
class TrainInstance(Instance):
	def __init__(self):
		super(TrainInstance, self).__init__()
		#STRUCT_LABEL_TRUE or STRUCT_LABEL_FALSE
		self.label = ''
		#True or False, whether the instance is above segment level in the corpus
		self.segment_flag = ''
		#for processing segments with lstm if needed
		self.segment_spans = []
		#source corpus id + '-train' + '-[produced order number]' + '-[instance number after sampling]'
		#ex: 001.xml-1-train-5-2 
		self.id = ''
		
#test instance for nn model
class TestInstance(Instance):
	def __init__(self):
		super(TestInstance, self).__init__()
		self.segment_spans = []
		self.edu_spans = []
		self.du_i_relations = []
		self.seg_i_relations = []
		#list of list, word level relations for each segments
		self.w_i_relations_list = []
		#ex:[u'，', u'。'], end puncs for each segment
		self.puncs = []
		#source corpus id + '-test' + '-[produced order number]' + '-[instance number after sampling]'
		#ex: 001.xml-1-train-5-2 
		self.id = ''

#relation in Instance
class I_Relation():
	def __init__(self, spans, sense, center, type):
		# ex: [[0, 46], [47, 76]]
		self.spans = spans
		#Variable int
		self.sense = sense
		#Variable int
		self.center = center
		#connective type, utf-8, explicit or implicit
		self.type = type
		
#read xml files from given directory and get Corpus list
def xml_dir_to_corpus_list(xml_dir):
	corpus_list = []
	
	#for parsing xml file, get the parsed tree root 
	#tutorial: http://t.cn/RAskmKa
	utf8_parser = ET.XMLParser(encoding='utf-8')
	
	for filename in os.listdir(xml_dir):					
		if os.path.splitext(filename)[-1] == '.xml':			
			file_path = os.path.join(xml_dir,filename)
			#get the parsing tree of xml file
			tree = ET.parse(file_path)
			root = tree.getroot()
			filename = os.path.basename(file_path)
			#for corpus id
			paragraph_count=1
			for p in root.iter('P'):
				rows = p.findall('R')
				# some paragraph may be empty!
				if rows == []:
					continue
				corp = Corpus()
				r = rows[0]
				corp.text = r.get('Sentence').replace(u'|', '')	
				
				corp.segment_spans = text_to_segment_spans(corp.text)
				#get the relation object list from xml rows
				#the xml rows list the relations in top down, left first style
				corp.du_relations = xml_rows_to_ralations(rows)
				#may need to assure the pre order of the relations
				#there is indeed some paragraph not pre-order from xml!
				corp.du_relations = relations_to_pre_order(corp.du_relations)
				#get the EDU index by just checking the kinds of the boundary in the relations
				corp.edu_spans = relations_to_edu_spans(corp.du_relations)
				
				corp.id = filename+'-'+str(paragraph_count)
				paragraph_count += 1				
				#print_corpus(corp)
				#check whether the edu span and segment span are coodinated
				corp.span_certification()
				
				corpus_list.append(corp)
	return corpus_list

# a parseseg_dict is indexed by file id number. 
def parse_dir_to_parseseg_dict(parse_dir):
	#for syntactic tree
	from nltk.tree import Tree
	utf8_parser = ET.XMLParser(encoding='utf-8')
	#for each file in the directory, use file_id to index
	parseseg_dict = {}
	for filename in os.listdir(parse_dir):
		#make sure the files are what we want
		#print filename
		if filename[-3:] == '.nw':
			#for parseseg id
			file_id = int(filename.split('.')[0].split('_')[-1])
			parseseg_dict[file_id] = [] 
			file_path = os.path.join(parse_dir,filename)
			et_tree = ET.parse( file_path)
			root = et_tree.getroot()
			#for id, count from 1
			segment_count = 1
			for sentence in root.iter('S'):
				p_tree = Tree.fromstring(sentence.text)
				
				#return relations, word_count, text	
				#make relations of a parsing tree in postfixed order
				relations, _,  words, text = tree_to_relations_and_words(p_tree, 0)			
				spans = text_to_segment_spans(text)
				#split relations to several part according to the spans
				splitted_relations = split_relations_by_spans(relations, spans)
				word_idx = 0
				span_idx = 0
				for relations in splitted_relations:
					word_start_idx = word_idx
					while words[word_idx].span[-1] <= spans[span_idx][-1]:
						word_idx += 1
						if word_idx >= len(words):
							break
					seg_words = words[word_start_idx:word_idx] 
					t = text[ spans[span_idx][0] : spans[span_idx][-1]+1 ]
					rs = relations_to_pre_order(relations)
					id = filename+'-'+str(segment_count)
					segment_count += 1
					span_idx += 1
					parseseg = ParseSeg(t, rs, seg_words, id)
					parseseg_dict[file_id].append(parseseg)
	return parseseg_dict	

def text_to_test_instance(text):
	instance = TestInstance()
	instance.segment_spans = text_to_segment_spans(text)
	instance.fragments = [text[i] for i in range(len(text))]
	instance.fragment_spans = [[i,i] for i in range(len(text))]
	for span in instance.segment_spans:
		instance.puncs.append( text[span[-1]] )
	return instance
	
#the instance.id of output instances is the same, ex: ex: 001.xml-1-train-5-5
def corpus_to_train_instance_list(corpus):
	# for finding pairs, we only store the span, not the whole Relation object
	#after find all neighboring pairs, we recover the relation structure and make instances		
	tr_instances = []
	#use a copy of the corpus to convert to binary structure, not the original object
	corp = copy.deepcopy(corpus)
	
	relations = corp.du_relations
	relations.extend(corp.seg_relations)
	relations.extend(corp.w_relations)
	
	#make relations binary preorder
	relations = relations_to_binary_preorder(relations)
	#the key of the span_to_relation_structure_dict is a span, ex: [ [0,5],[6,10] ]
	#the value is the pre order relations when see the key span as root
	spans_to_relation_structure_dict = get_spans_to_relation_structure_dict(relations)
	#spans information of words level under each segments
	
	
	w_spans_list_list = []
	for i in range(len(corp.segment_spans)):
		w_spans_list_list.append([])
	#spans information above segment level
	seg_du_spans_list = []
	#for w_spans_list_list,  make use of pre order
	seg_idx = 0
	
	for relation in relations:
		while relation.spans[0][0] > corp.segment_spans[seg_idx][-1]:
			seg_idx += 1

		# condition that the span is above segment level
		if relation.spans[0][0] == corp.segment_spans[seg_idx][0] and relation.spans[-1][-1] >= corp.segment_spans[seg_idx][-1]:
			seg_du_spans_list.append(relation.spans)
		# condition that the span is below segment level
		if relation.spans[-1][-1] <= corp.segment_spans[seg_idx][-1]:
			w_spans_list_list[seg_idx].append(relation.spans)
	#add each word span
	for seg_idx in range(len(corp.segment_spans)):	
		w_spans_list_list[seg_idx] = [ [[i,i]] for i in range(corp.segment_spans[seg_idx][0], corp.segment_spans[seg_idx][1]+1) ] + w_spans_list_list[seg_idx]
	
	#we separate the seg_du_spans_list and w_spans_list_list to avoid psedo relation construction across word level and segment/du level
	#we need seg_du_spans_list to be the first element of spans_list_list to use segment_flag
	spans_list_list = [seg_du_spans_list]
	if not seg_du_flag:
		spans_list_list.extend(w_spans_list_list)
	#for instance.id, count from 1
	instance_count = 1
	segment_flag = True
	for spans_list in spans_list_list:
		spans_list = spans_list_to_pre_order(spans_list)
		#print spans_list
		for i in range(len(spans_list)):
			now_spans = spans_list[i]
			#thanks for the pre order, we only consider the spans after now_span
			for j in range(i, len(spans_list)):
				# if the span is neighboring right to the now_span, then we can get an instance  
				if spans_list[j][0][0] == now_spans[-1][-1]+1:
					recovered_relations = recover_relations_from_spans_pair(now_spans, spans_list[j], spans_to_relation_structure_dict)
					tr_instance = get_train_instance_from_corpus_and_relations(corp, recovered_relations)
					#to post order, for construct structure when training
					tr_instance.i_relations = relations_to_post_order(tr_instance.i_relations)
					tr_instance.segment_flag = segment_flag
					#set id
					tr_instance.id = corp.id+'-train'+'-'+str(instance_count)+'-'+str(instance_count)
					tr_instances.append(tr_instance)
					instance_count += 1
		segment_flag = False
	return tr_instances

def corpus_to_test_instance(corpus, binary=True):
	corp = copy.deepcopy(corpus)
	te_instance = TestInstance()
	
	#make the relations binary pre-order and put them in the instance
	if binary:
		corp.du_relations = relations_to_binary_preorder(corp.du_relations)
		corp.seg_relations = relations_to_binary_preorder(corp.seg_relations)
		corp.w_relations = relations_to_binary_preorder(corp.w_relations)

	for relation in corp.du_relations:
		i_relation = I_Relation(relation.spans, SENSE_TO_LABEL[relation.sense], CENTER_TO_LABEL[relation.center], relation.type)
		te_instance.du_i_relations.append(i_relation)
	for relation in corp.seg_relations:
		i_relation = I_Relation(relation.spans, SENSE_TO_LABEL[relation.sense], CENTER_TO_LABEL[relation.center], relation.type)
		te_instance.seg_i_relations.append(i_relation)
	#separate each segment w_i_relations
	segment_count = 0
	te_instance.w_i_relations_list.append([])
	for relation in corp.w_relations:
		i_relation = I_Relation(relation.spans, SENSE_TO_LABEL[relation.sense], CENTER_TO_LABEL[relation.center], relation.type)
		if i_relation.spans[0][0] > corp.segment_spans[segment_count][-1]:
			segment_count += 1
			te_instance.w_i_relations_list.append([])
		te_instance.w_i_relations_list[-1].append(i_relation)

	
	te_instance.segment_spans = corp.segment_spans
	te_instance.edu_spans = corp.edu_spans
	te_instance.fragments = [corp.text[i] for i in range(len(corp.text))]
	te_instance.fragment_spans = [[i,i] for i in range(len(corp.text))]
	for span in corp.segment_spans:
		te_instance.puncs.append( corp.text[span[-1]] )
	get_words_list_in_instance_from_corpus(te_instance, corp)
	te_instance.id = corp.id + '-test' + '-1-1'
	
	return te_instance

#the key of the span_to_relation_structure_dict is a span, ex: [ [0,5],[6,10] ]
#the value is the pre order relations when see the key span as root
def get_spans_to_relation_structure_dict(relations):
	
	#[] when use edu span as key in the future
	spans_to_relation_structure_dict = defaultdict(list)
	
	
	#build span_to_relation_structure_dict
	for i in range(len(relations)):
		now_spans = relations[i].spans
		#thanks for the pre order, we only consider the spans 'now' and after now_span
		for j in range(i, len(relations)):
			# if the span is out of now_span
			if relations[j].spans[0][0] < now_spans[0][0] or relations[j].spans[-1][-1] > now_spans[-1][-1]: break
			#use str() to make list hashable
			spans_to_relation_structure_dict[str(now_spans)].append( relations[j] )

	return spans_to_relation_structure_dict
	
def recover_relations_from_spans_pair(spans_left, spans_right, spans_to_relation_structure_dict):
	
	#get the start boundary and end boundary of span_left and span_right 
	span_unit_left = [ spans_left[0][0], spans_left[-1][-1] ]
	span_unit_right = [ spans_right[0][0], spans_right[-1][-1] ]
	
	root_span = [ span_unit_left, span_unit_right ]
	
	relation_list = spans_to_relation_structure_dict[str(root_span)]
	
	# if in the dict, the root span doesn't exist, it means the relation is not true in the original structure.We construct a psuedo Relation for root relation where sense and center is None
	if relation_list == []:
		relation_list = spans_to_relation_structure_dict[str(spans_left)] + spans_to_relation_structure_dict[str(spans_right)]
		root_relation = Relation(root_span, PSEUDO_SENSE, PSEUDO_CENTER, NON_TYPE)
		relation_list = [root_relation] + relation_list

	return relation_list
	
def get_train_instance_from_corpus_and_relations(corpus, relations):
	
	tr_instance = TrainInstance()
	
	i_relations = []
	for relation in relations:
		i_relation = I_Relation(relation.spans, SENSE_TO_LABEL[relation.sense], CENTER_TO_LABEL[relation.center], relation.type)
		i_relations.append(i_relation)
	tr_instance.i_relations = i_relations	
	
	if i_relations[0].sense == SENSE_TO_LABEL[PSEUDO_SENSE]:
		tr_instance.label = STRUCT_LABEL_FALSE
	else:
		tr_instance.label = STRUCT_LABEL_TRUE
	
	start_idx = i_relations[0].spans[0][0]
	end_idx = i_relations[0].spans[-1][-1]
	
	tr_instance.fragment_spans = [[idx,idx] for idx in range(start_idx,end_idx+1)]
	tr_instance.fragments = [corpus.text[idx] for idx in range(start_idx,end_idx+1)]
	#for segment spans
	tr_instance.segment_spans = []
	for span in corpus.segment_spans:
		if span[0] < start_idx:
			continue
		if span[-1] > end_idx:
			break
		tr_instance.segment_spans.append(span)
	#for words
	get_words_list_in_instance_from_corpus(tr_instance, corpus)
		
	
	
	return tr_instance

def test_instance_and_text_to_dict(test_instance,text):
	dict = relations_and_text_to_dict(test_instance.du_i_relations, text)
	return dict
	
def relations_and_text_to_dict(relations,text):
	dict = {}
	now_relation = relations[0]
	dict['sense'] = LABEL_TO_SENSE[now_relation.sense]
	dict['center'] = LABEL_TO_CENTER[now_relation.center]
	dict['args'] = []
	arg_count = 0
	rel_idx = 1
	start_idx = 1
	for now_span in now_relation.spans:
		while rel_idx < len(relations) and relations[rel_idx].spans[0][0] <= now_span[-1]:
			rel_idx += 1
		#detect a leaf node
		if start_idx == rel_idx:
			dict['args'].append(text[now_span[0]:now_span[-1]+1].encode('utf-8'))
			continue
		else:
			dict['args'].append(relations_and_text_to_dict(relations[start_idx:rel_idx],text))
			start_idx = rel_idx
	return dict
			
				
	
def get_words_list_in_instance_from_corpus(instance, corpus):
	instance.words_list = []
	word_i = 0
	for span in instance.segment_spans:
		words = []
		word = corpus.words[word_i]
		seg_start_idx = span[0]
		seg_end_idx = span[-1]
		#we need to consider the cross boundary word like '「熬」到 '
		while word.span[0] <= seg_end_idx:
			if word.span[0] < seg_start_idx:
				if word.span[-1] >= seg_start_idx:
					#print corpus.id
					#print_word(word, corpus.text)
					new_word = Word([seg_start_idx, word.span[-1]],word.pos)
					#print_word(new_word, corpus.text)
					words.append(new_word)
				word_i += 1
			elif word.span[-1] > seg_end_idx:
				if word.span[0] <= seg_end_idx:
					#print corpus.id
					#print_word(word, corpus.text)
					new_word = Word([word.span[0], seg_end_idx],word.pos)
					#print_word(new_word, corpus.text)
					words.append(new_word)			
				#we don't do word_i += 1 because the same word might be used in latter segment
				break
			else:
				words.append(word)
				word_i += 1
			
			if word_i >= len(corpus.words):
				break
			word = corpus.words[word_i]

		instance.words_list.append(words)
		if word_i >= len(corpus.words):
			break

# add segment level  relation	
def add_corpus_seg_relations(corpus):
	seg_relations = []
	seg_i = 0
	for edu_span in corpus.edu_spans:
		#initial blank left node
		left_span = []
		#while still in the edu
		while corpus.segment_spans[seg_i][-1] <= edu_span[-1]:
			if left_span == []:
				left_span = corpus.segment_spans[seg_i]
			else:
				right_span = corpus.segment_spans[seg_i]
				# a edu is finished
				if right_span[-1] == edu_span[-1]:
					sense = EDU_SENSE
				else:
					sense = PRE_EDU_SENSE
				relation = Relation([left_span,right_span], sense, NON_CENTER, NON_TYPE)
				seg_relations.append(relation)
				# make a new left node span
				left_span = [ left_span[0], right_span[-1] ]
			# if it is the last segment
			if seg_i + 1 == len(corpus.segment_spans):
				break
				
			seg_i += 1
	corpus.seg_relations = seg_relations
	return corpus	

#merge the parseseg relations of each segment to a Corpus in the same file id number	
def merge_parseseg_dict_to_corpus_list(corpus_list, parseseg_dict): 
	# start from a non existing file id
	file_id = 0
	parseseg_idx = 0
	not_found = 0

	for corpus in corpus_list:
		#print corpus.id
		now_file_id = int(corpus.id.split('.')[0].split('-')[-1])
		if now_file_id != file_id:
			file_id = now_file_id
			parseseg_idx = 0
		parseseg_list = parseseg_dict[file_id]
		for span_idx in range(len(corpus.segment_spans)):
			seg_spans = corpus.segment_spans[span_idx]
			seg_text = corpus.text[seg_spans[0] : seg_spans[-1]+1]
			parseseg = parseseg_list[parseseg_idx]
			while seg_text != parseseg.text:
				#print seg_text.encode('utf-8')
				#print parseseg.text.encode('utf-8')
				#print '\n'
				parseseg_idx += 1
				#assuming we can find a match before out of boundary
				if parseseg_idx >= len(parseseg_list):
					print corpus.id
					print seg_text.encode('utf-8')
					for pseg in parseseg_list:
						print pseg.text.encode('utf-8')
					not_found += 1
					parseseg_idx = 0
					break
				parseseg = parseseg_list[parseseg_idx]
			
			#modify parseg span to corpus segment span
			#print_parseseg(parseseg)
			modify_parseg_span_to_segment_spans(parseseg, seg_spans)
			corpus.words.extend(parseseg.words)
			corpus.w_relations.extend(parseseg.relations)
			#break
		
		#for set the EDU_SENSE
		edu_idx = 0
		#there may be many relations suit one EDU since it's not binary
		for relation in corpus.w_relations:
			if relation.spans[0][0] > corpus.edu_spans[edu_idx][-1]:
				edu_idx += 1
			if relation.spans[0][0] == corpus.edu_spans[edu_idx][0] and relation.spans[-1][-1] == corpus.edu_spans[edu_idx][-1]:
				relation.sense = EDU_SENSE
			if edu_idx == len(corpus.edu_spans):
				break
	print 'not_found count:', not_found	
	return corpus_list	

def relations_binary_to_multi_preorder(relations):
		#reorganize to pre-order
		relations = relations_to_pre_order(relations)
		new_relations = []
		r_idx = 0
		#since relations is preorder, we can find left child(which is not EDU) by checking next relation
		if len(relations) > 0:
			new_relations.append(relations[0])
		while r_idx < len(relations)-1:
			r = new_relations.pop()
			next_r = relations[r_idx+1]
			#the relation now is a coordination and equal relation 
			if r.sense == COORD_SENSE_LABEL and r.center == COORD_CENTER_LABEL:
				print 'find'
				print_i_relation(r)
				print_i_relation(next_r)
				#the next relation is the left child of now relation a coordination and equal relation
				if next_r.spans[0][0] == r.spans[0][0] and next_r.sense == COORD_SENSE_LABEL and next_r.center == COORD_CENTER_LABEL:
					r.spans = next_r.spans+r.spans[1:]
					r_idx += 1
					new_relations.append(r)
					continue
			new_relations.append(r)
			new_relations.append(next_r)
			r_idx += 1
		
		return new_relations
	
def relations_to_binary_preorder(relations):
		new_added_relations = []
		to_be_removed_relations = []
		for relation in relations:
			if len(relation.spans) > 2:
				#to delete the original multi-children relation in the future
				to_be_removed_relations.append(relation)
				#ex: [0,3]
				left_span_unit = relation.spans[0]
				for idx in range(1,len(relation.spans)):
					right_span_unit = relation.spans[idx]
					new_spans = [left_span_unit, right_span_unit]
					new_relation = Relation(new_spans, relation.sense, relation.center, relation.type)
					new_added_relations.append(new_relation)
					left_span_unit = [ new_spans[0][0], new_spans[1][1] ]
			elif len(relation.spans) == 1:
				to_be_removed_relations.append(relation)
		for r in to_be_removed_relations:
			relations.remove(r)
		for r in new_added_relations:
			relations.append(r)
		
		#reorganize to pre-order
		relations = relations_to_pre_order(relations)
		
		return relations

	
def build_word_to_ix_from_corpus_list(corpus_list):	
	
	for corpus in corpus_list:
		for word in corpus.text:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
		#for idx of oov
		oov = len(word_to_ix)
		word_to_ix['oov'] = len(word_to_ix) #for oov when test

#dump the dictionary to a file		
def save_word_to_ix():
	global word_to_ix
	with open(WORD_TO_IX_DUMP_PATH, "wb") as myFile:
		pickle.dump(word_to_ix, myFile)	
		
def load_word_to_ix():
	global word_to_ix
	with open(WORD_TO_IX_DUMP_PATH, "rb") as myFile:
		word_to_ix = pickle.load(myFile)	
	
#to judge the '——', '。」' condition
def is_punc_in_text(text, idx):	
	if text[idx] in _ENDs:
		if idx+1 < len(text):
			if text[idx+1] in _ENDs:
				if text[idx] == u'—' or text[idx+1] != u'—':
					return False
		if text[idx] == u'—' and idx > 1:
			if text[idx-1] != u'—':
				return False
		return True
	else:
		return False

	
#get the relation object list from xml rows
def xml_rows_to_ralations(xml_rows):
		relations = []

		for r in xml_rows:
			sense = r.get('RelationType')
			center = r.get('Center')
			#get the span
			spans = []
			#sentence_position ex: u'27…51|52…63|64…78|79…89'			
			sentence_position = r.get('SentencePosition')
			positions = sentence_position.split(u'|')
			for p in positions:
				ps = p.split(u'…')
				#the span index is start from 0, not 1, different with the xml format
				spans.append( [ int(ps[0])-1 ,int(ps[-1])-1 ] )
			relation = Relation(spans, sense, XML_TO_CORPUS_CERTER_DICT[center], r.get('ConnectiveType'))	
			relations.append(relation)
			
		return relations

#get the EDU index by just checking the kinds of the boundary in the relations
def relations_to_edu_spans(relations):
	edu_spans = []		
	#this set used to collected all kinds of boundary
	boundary_idx_set = set()
	for relation in relations:
		#span example: [ [0, 21], [22, 76], [77, 129] ]
		for span_unit in relation.spans:
			boundary_idx_set.add(span_unit[0])
			boundary_idx_set.add(span_unit[1])

	#sorted from left to right
	boundary_idx = sorted(boundary_idx_set)

	for i in range(len(boundary_idx)-1):
		# 2 index is a pair
		if i%2 == 0:
			edu_spans.append([ boundary_idx[i], boundary_idx[i+1] ])
			
	return edu_spans

#get segments idx, use punctuations split the text to segments
def text_to_segment_spans(text):
	start = 0
	end = 0
	segment_spans = []

	for idx in range(len(text)):
		if is_punc_in_text(text, idx):
			end = idx
			segment_spans.append([start,end])
			start = end+1
		
	#some corpus has no punc in the end!!
	if start != len(text):
		segment_spans.append([start,len(text)-1])
		
	return segment_spans
	
	
def modify_parseg_span_to_segment_spans(parseseg, seg_spans):
	shift_len = seg_spans[0] - parseseg.relations[0].spans[0][0]
	#print 'shift_len:', shift_len
	for relation in parseseg.relations:
		new_spans = []
		for sp in relation.spans:
			new_spans.append( [sp[0]+shift_len, sp[-1]+shift_len] )
		relation.spans = new_spans
	for word in parseseg.words:
		new_span = [ word.span[0]+shift_len, word.span[-1]+shift_len ]
		word.span = new_span
		
	return 
		
	
#split relations to several part according to the spans
def split_relations_by_spans(relations, spans):
	splitted_relations = []
	for i in range(len(spans)):
		splitted_relations.append([])
	#for simplest condition
	if len(spans) == 1:
		splitted_relations[0].extend(relations)
		return splitted_relations
	else:
		span_idx = 0
		for relation in relations:
			r_spans = relation.spans
			now_span = spans[span_idx]
			# make now_span catch up with the beginning of r_spans
			while r_spans[0][0] > now_span[-1]:
				span_idx += 1
				now_span = spans[span_idx]
			# if now_span fully cover r_spans
			if r_spans[-1][-1] <= now_span[-1] and r_spans[0][0] >= now_span[0]:
				splitted_relations[span_idx].append(relation)
			# need multiple spans to cover r_spans
			else:
				#print 'hard condition'
				span_idx = 0
				now_span = spans[span_idx]
				# make now_span catch up with the beginning of r_spans
				while r_spans[0][0] > now_span[-1]:
					span_idx += 1
					now_span = spans[span_idx]				
				#the relation spans in the current relation of the relation of now_span
				sub_spans = []
				for r_s in r_spans:
					#make a copy to operate
					r_span = r_s[:]
					#while r_span hasn't been used out
					while r_span != []:
						#print sub_spans, r_span
						#if now_span has not caught up with r_span
						if r_span[0] > now_span[-1]:
							now_relation = Relation(sub_spans, PRE_EDU_SENSE, NON_CENTER, NON_TYPE)
							splitted_relations[span_idx].append(now_relation)
							span_idx += 1
							now_span = spans[span_idx]
							sub_spans = []
						else:
							#the inter-cover part
							sub_span = [ max(r_span[0],now_span[0]), min(r_span[-1], now_span[-1]) ]
							sub_spans.append(sub_span)
							#cut r_span
							if now_span[-1] >= r_span[-1]:
								r_span = []
							else:
								r_span = [now_span[-1]+1, r_span[-1]]
				#for the condition a now_span right inter-cover the r_spans
				if sub_spans != []:
					now_relation = Relation(sub_spans, PRE_EDU_SENSE, NON_CENTER, NON_TYPE)
					splitted_relations[span_idx].append(now_relation)
	return splitted_relations

#return relations, word_count, text	
#make relations of a parsing tree in postfixed order		
def tree_to_relations_and_words(p_tree, char_count):
	relations = []
	words = []
	text = ''
	#check if we reach the leaf
	if type(p_tree) == type(u''):
		chars = p_tree
		#print chars.encode('utf-8')
		n_char = len(chars)
		spans = []
		for i in range(n_char):
			span = [char_count,char_count]
			char_count += 1
			spans.append(span)
		relation = Relation(spans, PRE_EDU_SENSE, NON_CENTER, NON_TYPE)
		relations.append(relation)
		text = chars
		return relations,char_count, words, text
	else:
		spans = []
		#for '-NONE-' condition in the parsing tree
		if p_tree.label() == '-NONE-':
			return None, char_count, words, text
		for child_idx in range(len(p_tree)):
			#if we are about to reach the leaf, extract word information in advance
			if type(p_tree[child_idx]) == type(u''):
				pos = p_tree.label()
				n_char = len(p_tree[child_idx])
				w_span = [char_count, char_count+n_char-1]
				word = Word(w_span, pos)
				#print_word(word, text)
				words.append(word)
			sub_relations, char_count, sub_words, sub_text = tree_to_relations_and_words(p_tree[child_idx], char_count)
			if sub_relations == None:
				continue
			span = [ sub_relations[-1].spans[0][0], sub_relations[-1].spans[-1][-1] ]
			spans.append(span)
			relations.extend(sub_relations)
			words.extend(sub_words)
			text += sub_text
		if spans == []:
			return None, char_count, words, text
		relation = Relation(spans, PRE_EDU_SENSE, NON_CENTER, NON_TYPE)
		relations.append(relation)
		return relations, char_count, words, text

def spans_list_to_pre_order(spans_list):
	# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
	from functools import cmp_to_key
	#cmp_for_pre_order_from_span: compare function for sorting for pre order
	key_for_pre_order_from_spans = cmp_to_key(cmp_for_pre_order_from_spans)
		
	spans_list = sorted(spans_list, key=key_for_pre_order_from_spans, reverse=True)
		
	return spans_list
		
#compare function for sorting for pre order
def cmp_for_pre_order_from_spans(spans_1, spans_2):
	#compare the start boundary first
	if spans_1[0][0] < spans_2[0][0]:
		return 1
		#if same, compare the end boundary
	elif spans_1[0][0] == spans_2[0][0]:
		if spans_1[-1][-1] > spans_2[-1][-1]:
			return 1
	return -1		
		
def relations_to_pre_order(relations):
	# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
	from functools import cmp_to_key
	#cmp_for_pre_order: compare function for sorting for pre order
	key_for_pre_order_from_relation = cmp_to_key(cmp_for_pre_order_from_relation)
		
	relations = sorted(relations, key=key_for_pre_order_from_relation, reverse=True)
		
	return relations

#compare function for sorting for pre order
def cmp_for_pre_order_from_relation(rel_1, rel_2):
	#compare the start boundary first
	if rel_1.spans[0][0] < rel_2.spans[0][0]:
		return 1
		#if same, compare the end boundary
	elif rel_1.spans[0][0] == rel_2.spans[0][0]:
		if rel_1.spans[-1][-1] > rel_2.spans[-1][-1]:
			return 1
	return -1

def relations_to_post_order(relations):
	# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
	from functools import cmp_to_key
	#cmp_for_post_order_from_i_relation: compare function for sorting for post order
	key_for_post_order_from_relation = cmp_to_key(cmp_for_post_order_from_relation)
		
	relations = sorted(relations, key=key_for_post_order_from_relation, reverse=True)
		
	return relations
	
#compare function for sorting for post order
def cmp_for_post_order_from_relation(r_1, r_2):
	#compare the end boundary first
	if r_1.spans[-1][-1] < r_2.spans[-1][-1]:
		return 1
		#if same, compare the start boundary
	elif r_1.spans[-1][-1] == r_2.spans[-1][-1]:
		if r_1.spans[0][0] > r_2.spans[0][0]:
			return 1
	return -1	

def print_EDUs_from_corpus(corpus):
	for span in corpus.edu_spans:
		print corpus.text[span[0]:span[-1]+1].encode('utf-8')
	
def print_corpus(corpus):
		print 'id:', corpus.id 	
		print 'text:', corpus.text.encode('utf-8')
		for word in corpus.words:
			print_word(word, corpus.text)
		print 'segments_span:', corpus.segment_spans
		print 'edus_span:', corpus.edu_spans
		print 'du_relations:'
		for relation in corpus.du_relations:
			print_relation(relation)
		print 'seg_relations:'
		for relation in corpus.seg_relations:
			print_relation(relation)
		print 'w_relations:'
		for relation in corpus.w_relations:
			print_relation(relation)
		
def print_word(word, text):
	if text != '':
		print 'word:', text[word.span[0]: word.span[-1]+1].encode('utf-8'),
	print 'span:',word.span,
	print 'pos:', word.pos.encode('utf-8'),
			
def print_relation(relation):
		print relation.spans, relation.sense.encode('utf-8'), relation.center.encode('utf-8'),relation.type.encode('utf-8'),
		return

def print_i_relation(relation):
	#if relation.sense != 5:
		print relation.spans, relation.sense, relation.center,
		return		

def print_parseseg(parseseg):
	print parseseg.id
	print parseseg.text.encode('utf-8')
	for word in parseseg.words:
		#text = '' due to index is not modified
		print_word(word, '')	
	for relation in parseseg.relations:
		print_relation(relation)
		
def print_train_instance(instance):
	for words in instance.words_list:
		for word in words:
			print_word(word, ''),
		print ''
	print 'fragments:'
	if type(instance.fragments[0]) == type(u''):
		for fragment in instance.fragments:
			print fragment.encode('utf8'),',',
	else:
		for fragment in instance.fragments:
			print fragment,',',		
	print ''
	print 'fragment_spans: ', instance.fragment_spans
	#ex:[[0, 46], [47, 61], [62, 76]]
	print 'segment_spans:', instance.segment_spans
	print 'i_relations:'
	for i_relation in instance.i_relations:
		print_i_relation(i_relation)
	print ''
	#True or False
	print'label:', instance.label
	print'id:', instance.id

def print_test_instance(instance):
	for words in instance.words_list:
		for word in words:
			print_word(word, ''),	
	print 'fragments:'
	if type(instance.fragments[0]) == type(u''):
		for fragment in instance.fragments:
			print fragment.encode('utf8'),',',
	else:
		for fragment in instance.fragments:
			print fragment,',',		
	print ''
	print 'fragment_spans: ', instance.fragment_spans
	print 'segment_spans:', instance.segment_spans
	print 'edu_spans:', instance.edu_spans	
	print 'puncs:'
	for punc in instance.puncs:
		print punc.encode('utf-8'),' ',
	#ex:[[0, 46], [47, 61], [62, 76]]
	print 'du_i_relations:'
	for i_relation in instance.du_i_relations:
		print_i_relation(i_relation)
	print 'seg_i_relations:'
	for i_relation in instance.seg_i_relations:
		print_i_relation(i_relation)
	print 'w_i_relations_list:'
	for i_relation_list in instance.w_i_relations_list:
		for i_relation in i_relation_list:
			print_i_relation(i_relation)
	print 'i_relations:' 
	for i_relation in instance.i_relations:
		print_i_relation(i_relation)	
	print ''
	print'id:', instance.id

	
def print_word_to_ix():
	for k, v in word_to_ix.iteritems():
		print k.encode('utf-8'),':',v,'\t',

