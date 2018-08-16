import data_structure
import pickle
import evaluate
import random

import argparse

all_xml_dir = 'raw_data/CDTB_data_repair'
xml_dir = 'raw_data/train_repair'
parse_dir = 'raw_data/train_parsed_repair'
test_xml_dir = 'raw_data/test_repair'
test_parse_dir = 'raw_data/test_parsed_repair'
train_corpus_list_file = './train_corpus_list_file'
train_instances_file = './train_instances_file'
test_corpus_list_file = './test_corpus_list_file'




def process_commands():
	parser = argparse.ArgumentParser()
	parser.add_argument('-demo','--demonstration', action='store_true',
						help='demo mode')
	parser.add_argument('-i', '--input_file',
						help='input file')
	parser.add_argument('-o', '--output_file', 
						help='output file')
	return parser.parse_args()
	
def demo():
	import model_user
	import json
	print 'demo mode'
	model = model_user.get_model('./model_rvnn_lstm_20')
	with open(args.input_file, "rb") as myFile:
		lines = myFile.readlines()
	text = ''
	for line in lines:
		text += line
	data_structure.load_word_to_ix()
	text = text.decode('utf-8')
	instance = data_structure.text_to_test_instance(text)
	return_instance = model_user.demo_predict(model, instance)
	output_data = {}
	output_data['EDUs'] = []
	output_data['relations'] = []
	edu_boundaries = set()
	for relation in return_instance.du_i_relations:
		relation_info = {}
		arg_count = 1
		for span in relation.spans:
			edu_boundaries.add(span[-1])
			arg = text[span[0]:span[-1]+1].encode('utf-8')
			relation_info['arg'+str(arg_count)] = arg
			arg_count += 1
		relation_info['sense'] = data_structure.LABEL_TO_SENSE[relation.sense]
		relation_info['center'] = data_structure.LABEL_TO_CENTER[relation.center]
		output_data['relations'].append(relation_info)
	edu_boundary_list = sorted(edu_boundaries)
	start_idx = 0
	for boundary in edu_boundary_list:
		output_data['EDUs'].append(text[start_idx:boundary+1].encode('utf-8'))
		start_idx = boundary+1
	dict = data_structure.test_instance_and_text_to_dict(return_instance, text)
	
	output_data['tree'] = dict
	with open(args.output_file, 'w') as outfile:  
		json.dump(output_data, outfile, ensure_ascii=False)
	#data_structure.print_test_instance(return_instance)



	
def experiment():
	import model_user
	'''
	# train corpus
	corpus_list = data_structure.xml_dir_to_corpus_list(xml_dir)	
	data_structure.build_word_to_ix_from_corpus_list(corpus_list)
	parseseg_dict = data_structure.parse_dir_to_parseseg_dict(parse_dir)
	corpus_list = corpus_list[:-1]
	for corpus in corpus_list:
		corpus = data_structure.add_corpus_seg_relations(corpus)
	corpus_list = data_structure.merge_parseseg_dict_to_corpus_list(corpus_list, parseseg_dict)

	#test_corpus
	test_corpus_list = data_structure.xml_dir_to_corpus_list(test_xml_dir)
	test_parseseg_dict = data_structure.parse_dir_to_parseseg_dict(test_parse_dir)
	for corpus in test_corpus_list:
		corpus = data_structure.add_corpus_seg_relations(corpus)
	test_corpus_list =\
	data_structure.merge_parseseg_dict_to_corpus_list(test_corpus_list, test_parseseg_dict)	

	
	data_structure.load_word_to_ix()

	corpus_list = corpus_list[:-1]
	#train instances
	instances = []
	for corpus in corpus_list:
		#if corpus.id == '246.xml-2':
		#	data_structure.print_corpus(corpus)
		instance_list = data_structure.corpus_to_train_instance_list(corpus)
		instances.extend(instance_list)
	

	#dump data to file
	with open(train_corpus_list_file, "wb") as myFile:
		pickle.dump(corpus_list, myFile)
	with open(test_corpus_list_file, "wb") as myFile:
		pickle.dump(test_corpus_list, myFile)
	with open(train_instances_file, "wb") as myFile:
		pickle.dump(instances, myFile)
	data_structure.save_word_to_ix()
	'''
	
	#load data
	#data_structure.load_word_to_ix()
	'''
	with open(train_corpus_list_file, "rb") as myFile:
		corpus_list = pickle.load(myFile)	
	
	for corpus in corpus_list:
		if corpus.id == '593.xml-2':
			data_structure.print_corpus(corpus)
	'''
	'''
	with open(train_instances_file, "rb") as myFile:
		instances = pickle.load(myFile)
	#instances = instances[:1]
	instances = random.sample(instances, 1000)
	'''
	print 'hellow'
	data_structure.load_word_to_ix()
	
	#train
	#model = model_user.get_model(model_path=None)
	#model_user.train_from_instances(model, instances)
	#test
	
	with open(test_corpus_list_file, "rb") as myFile:
		test_corpus_list = pickle.load(myFile)
	model = model_user.get_model('./model_rvnn_lstm_20')
	model_user.test_from_corpus_list(model, test_corpus_list)
	
	
def analysis():
	
	corpus_list = data_structure.xml_dir_to_corpus_list(all_xml_dir)
	'''
	for corpus in corpus_list:
		for span in corpus.edu_spans:
			print corpus.text[span[0]:span[-1]+1].encode('utf-8')
	'''
	with open(train_corpus_list_file, "rb") as myFile:
		corpus_list = pickle.load(myFile)
	with open(test_corpus_list_file, "rb") as myFile:
		test_corpus_list = pickle.load(myFile)
	evaluater = evaluate.Evaluater()
	evaluater.show_relation_distribution_from_corpus_list(corpus_list)		
	#evaluater.show_edu_punc_distribution_from_corpus_list(corpus_list)
	#evaluater.show_analysis_from_corpus_list(corpus_list)
	
	#corpus_list.extend(test_corpus_list)
	#evaluater.show_pos_distribution_from_corpus_list(corpus_list)
def test():	
	'''
	# train corpus
	corpus_list = data_structure.xml_dir_to_corpus_list(xml_dir)	
	parseseg_dict = data_structure.parse_dir_to_parseseg_dict(parse_dir)
	corpus_list = corpus_list[:-1]
	for corpus in corpus_list:
		corpus = data_structure.add_corpus_seg_relations(corpus)
	corpus_list = data_structure.merge_parseseg_dict_to_corpus_list(corpus_list, parseseg_dict)

	#test_corpus
	test_corpus_list = data_structure.xml_dir_to_corpus_list(test_xml_dir)
	test_parseseg_dict = data_structure.parse_dir_to_parseseg_dict(test_parse_dir)
	for corpus in test_corpus_list:
		corpus = data_structure.add_corpus_seg_relations(corpus)
	test_corpus_list =\
	data_structure.merge_parseseg_dict_to_corpus_list(test_corpus_list, test_parseseg_dict)	
	
	with open(train_corpus_list_file, "wb") as myFile:
		pickle.dump(corpus_list, myFile)
	with open(test_corpus_list_file, "wb") as myFile:
		pickle.dump(test_corpus_list, myFile)
	'''
	
	data_structure.load_word_to_ix()
	with open(train_corpus_list_file, "rb") as myFile:
		corpus_list = pickle.load(myFile)
	corpus_list = corpus_list[:-1]

	with open(test_corpus_list_file, "rb") as myFile:
		test_corpus_list = pickle.load(myFile)
	
	
	
	
	instances = []
	for corpus in corpus_list:
		instance_list = data_structure.corpus_to_train_instance_list(corpus)
		#for instance in instance_list:
			#data_structure.print_train_instance(instance)
		instances.extend(instance_list)

	
	test_instances = []
	for corpus in test_corpus_list:	
		test_instance = data_structure.corpus_to_test_instance(corpus, binary=True)
		test_instances.append(test_instance)
	
	

	with open(train_instances_file, "wb") as myFile:
		pickle.dump(instances, myFile)


args = process_commands()	
def main():
	
	if args.demonstration:
		demo()
	#test()
	#experiment()
	#analysis()
	
	
if __name__ == "__main__":
	main()