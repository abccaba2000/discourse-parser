'''functions for preprocessing'''

# the Reporter class is use to help log, statistics, debug, measure process time, an instance reporter is initialized when first import
import report
#to get arguments from shell 
#execute and stored the command arguments in variable args when first import
import args
#report.reporter and args.args are the two global variables shared by all files that import report.py and args.py

import corpus


# for parsing xml file
import xml.etree.ElementTree as ET
# for dealing with file system
import os

#get a list of Corpus objects defined in corpus.py, each object corresponds to a paragraph
#path is the corpus directory
def get_corpus_list(path):

	#log what we are doing, format: "<parm1> [from <src>]..."
	report.reporter.log('getting corpus', src=path)
	#set the start time, for measure process time
	report.reporter.set_time()
	
	#initilize the list about to be returned
	corpus_list = []
	
	#the case the corpus resource is of xml format
	if args.args.xml:
		#for each file in the directory
		for filename in os.listdir(path):					
			#filter '.xml' file
			if os.path.splitext(filename)[-1] == '.xml':			
				#log what file we are processing, reusing a line when called repeatedly 
				report.reporter.log('now processing {}'.format(filename), repeat=True )
				
				file_path = os.path.join(path,filename)
				#for getting Corpus object list from a xml file
				#a file contains several paragraphs, so does several Corpus object
				c_list = get_corpus_from_xml(file_path)
				corpus_list.extend(c_list)
			
			else: continue
			
				
	#built vocabulary
	corpus.build_word_to_ix(corpus_list)
	
			
				
				
	#log the process time
	report.reporter.measure_time()
	
	return corpus_list
	
#for getting Corpus object list defined in corpus.py from xml file
#a file contains several paragraphs, so does several Corpus object
#path is the xml file path

def get_corpus_from_xml(path):

	#for parsing xml file, get the parsed tree root 
	#tutorial: http://t.cn/RAskmKa
	utf8_parser = ET.XMLParser(encoding='utf-8')
	tree = ET.parse(path)
	root = tree.getroot()
	
	#initilize the list about to be returned
	c_list = []
	
	
	for p in root.iter('P'):
		rows = p.findall('R')
		# some paragraph may be empty!
		if rows == []:
			continue
		
		#initilize an empty Corpus
		corp = corpus.Corpus()
		#get a Corpus object from xml rows. The span index is start from 0, not 1, different with the xml format
		#in xml rows,  the relations are in top down, pre order. This style remains the same
		#may need to assure the pre order of the relations
		#there is indeed some paragraph not pre-order from xml!
		corp.xml_rows_to_corpus(rows)
		if args.args.paragraph != None:
			if corp.id != int(args.args.paragraph):
				continue
		
		
		#to output the object information, object_attr is for the attribute that itself is an object, for output nested object
		#report.reporter.show_attributes(corp, object_attr='relations')
		
		c_list.append(corp)
		
	
	
	return c_list
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		