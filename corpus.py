# coding=UTF-8
'''
define the Corpus class which contains information of a paragraph: text, segments index, relations and the merge structure

merge structure: segments merge to EDU, EDUS merge to bigger discourse unit 

'''
#to get arguments from shell 
#execute and stored the command arguments in variable args when first import
import args
# the Reporter class is use to help log, statistics, debug, measure process time, an instance reporter is initialized when first import
import report

#these punctuation divide text to segments
_ENDs = (u'?', u'”', u'…', u'──', u'、', u'。', u'」', u'！', u'，', u'：', u'；', u'？')

#every time a Corpus is initialized, we assign it an id, using global variable corpus_id_count to count the id
corpus_id_count = 0

word_to_ix = {}
oov = 0

def build_word_to_ix(corpus_list):	
	
	for corpus in corpus_list:
		for word in corpus.text:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
		#for idx of oov
		oov = len(word_to_ix)
		word_to_ix['oov'] = len(word_to_ix) #for oov when test

class Corpus():
	def __init__(self):
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing
		self.text = ''
		#each element is a [start_idx, end_idx] list, a text is devided to segments by given punctuations
		self.segments_span = []
		#each element is a [start_idx, end_idx] list, some segments merge to a EDU
		self.edus_span = []
		#top-down, pre order
		self.relations = []
		
		#every time a Corpus is initialized, we assign it an id, using global variable corpus_id_count to count the id
		global corpus_id_count
		self.id = corpus_id_count
		corpus_id_count += 1
	
	#get a Corpus object from xml rows. The span index is start from 0, not 1, different with the xml format
	#in xml rows,  the relations are in top down, pre order. This style remains the same
	def xml_rows_to_corpus(self, xml_rows):
		#each row is a relation
				
		#get the text from the first row
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing	
		r = xml_rows[0]
		self.text = r.get('Sentence').replace(u'|', '')
	
		#get segments idx, using the global _EDU tuple to split the text to segments
		self.segments_span = self.text_to_segments_span(self.text)

		#get the relation object list from xml rows
		#the xml rows list the relations in top down, left first style
		self.relations = self.xml_rows_to_ralations(xml_rows)
		#may need to assure the pre order of the relations
		#there is indeed some paragraph not pre-order from xml!
		self.relations = self.relations_to_pre_order(self.relations)

		#get the EDU index by just checking the kinds of the boundary in the relations
		self.edus_span = self.find_edus_span_from_relations(self.relations)
		

	#get segments idx, using the global _EDU tuple to split the text to segments		
	def text_to_segments_span(self, text):
		start = 0
		end = 0
		segments_span = []

		for idx in range(len(text)):			
			word = text[idx]
			if word in _ENDs:
				end = idx
				segments_span.append([start,end])
				start = end+1
		
		#some corpus has no punc in the end!!
		if start != len(text):
			segments_span.append([start,len(text)-1])
		
		return segments_span
				
	#get the relation object list from xml rows
	#the xml rows list the relations in top down, pre order
	def xml_rows_to_ralations(self, xml_rows):
		relations = []

		for r in xml_rows:
			#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing
			sense = r.get('RelationType')
			center = r.get('Center')
					
			#get the span
			span = []
			#sentence_position ex: u'27…51|52…63|64…78|79…89'			
			sentence_position = r.get('SentencePosition')
			positions = sentence_position.split(u'|')
			for p in positions:
				ps = p.split(u'…')
				#the span index is start from 0, not 1, different with the xml format
				span.append( [ int(ps[0])-1 ,int(ps[-1])-1 ] )
				
				
			relation = Relation(span, sense, center)	
			relations.append(relation)
			
		return relations
	
	#get the EDU index by just checking the kinds of the boundary in the relations
	def find_edus_span_from_relations(self, relations):
		edus_span = []
		
		#this set used to collected all kinds of boundary
		boundary_idx_set = set()
		for relation in relations:
			#span example: [ [0, 21], [22, 76], [77, 129] ]
			for span_unit in relation.span:
				boundary_idx_set.add(span_unit[0])
				boundary_idx_set.add(span_unit[1])

		#sorted from left to right
		boundary_idx = sorted(boundary_idx_set)

		for i in range(len(boundary_idx)-1):
			# 2 index is a pair
			if i%2 == 0:
				edus_span.append([ boundary_idx[i], boundary_idx[i+1] ])
			
		return edus_span
	
	def relations_to_pre_order(self, relations):
		# this is for python3 removing the cmp in the sorted function, to transfer a cmp to key
		from functools import cmp_to_key
		#cmp_for_pre_order: compare function for sorting for pre order
		key_for_pre_order_from_relation = cmp_to_key(self.cmp_for_pre_order_from_relation)
		
		relations = sorted(relations, key=key_for_pre_order_from_relation, reverse=True)
		
		return relations
		
	def to_binary_structure(self):
		relations = self.relations
		new_added_relations = []
		to_be_removed_relations = []
		for relation in relations:
			if len(relation.span) > 2:
				#to delete the original multi-children relation in the future
				to_be_removed_relations.append(relation)
				#ex: [0,3]
				left_span_unit = relation.span[0]
				for idx in range(1,len(relation.span)):
					right_span_unit = relation.span[idx]
					new_span = [left_span_unit, right_span_unit]
					new_relation = Relation(new_span, relation.sense, relation.center)
					new_added_relations.append(new_relation)
					left_span_unit = [ new_span[0][0], new_span[1][1] ]
		
		for r in to_be_removed_relations:
			relations.remove(r)
		for r in new_added_relations:
			relations.append(r)
		
		self.relations = self.relations_to_pre_order(relations)
		return
					
	
	#compare function for sorting for pre order
	def cmp_for_pre_order_from_relation(self, rel_1, rel_2):
		#compare the start boundary first
		if rel_1.span[0][0] < rel_2.span[0][0]:
			return 1
			#if same, compare the end boundary
		elif rel_1.span[0][0] == rel_2.span[0][0]:
			if rel_1.span[-1][-1] > rel_2.span[-1][-1]:
				return 1
		return -1
	

		
class Relation():
	def __init__(self, span, sense, center):
		#The span index is start from 0, not 1, different with the xml format
		self.span = span
		#encoding = unicode, easy  to index,  need to decode back to utf-8  for printing 
		self.sense = sense
		# 1, 2, or 3
		self.center = center
		return
		