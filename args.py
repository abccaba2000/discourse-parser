'''
	function process_commands():
	to get arguments from shell
	execute and stored the command arguments in args when initial import by parser_main.py 
'''
#for getting arguments from the shell
import argparse


'''	
	'--xxx' means option arguments,	if without the '---', 'xxx' means propositional argument, '-x' is the abbreviation of '--xxx'
	'store_true' means value is "True" when specified
	
	for tutorial: http://t.cn/REzeJTj
'''
def process_commands():
	parser = argparse.ArgumentParser()
 
	parser.add_argument('-x','--xml', action='store_false',
						help='whether to load data in xml format')
	parser.add_argument('-tr', '--train', action='store_true',
						help='whether to train the model')
 
	#set mutual exclusive argument using group
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-te1', '--one_test', action='store_true',
						help='whether to test a function by just processing one(the first) paragraph(in test stage)(and only use one instance for the corpus to train)')
	group.add_argument('-p', '--paragraph', default=None, type=int,
						help='examine a specific paragraph by giving the paragraph id(in test or train stage)')
	group.add_argument('-tr1', '--one_train', action='store_true',
						help='whether to test a function by just processing one(the first) paragraph(in train stage)(and only use one instance for the corpus to train)')
	#'./model_12epoch', './model_join_objective_15epoch'
	parser.add_argument('-sm', '--save_model', default='./model_double_lstm_20_epoch',
						help='whether to save the training model to the given model name')
	parser.add_argument('-lm', '--load_model', default='./model_double_lstm_20_epoch',
						help='whether to load the  model from the given model name')
	
	#training options
	parser.add_argument('-trre','--train_right_edu', action='store_true',
						help='use right first merge when considering segments merging to edu')	
	
	#predict options
	parser.add_argument('-pe','--punctuation_limit_edu', action='store_false',
						help='use some punctuation limit policy when predicting edu')
	parser.add_argument('-ge','--gold_edu', action='store_true',
						help='use gold EDU when predicting')
	parser.add_argument('-gex','--gold_edu_direct', action='store_true',
						help='use gold EDU and send the EDU to lstm directly when predicting')
	parser.add_argument('-sqe','--sequence_predict_edu', action='store_true',
						help='use sequence predict EDU method when predicting')
	parser.add_argument('-gre','--greedy_predict_edu', action='store_true',
						help='use greedy predict EDU method when predicting')
	parser.add_argument('-grs','--greedy_predict_structure', action='store_true',
						help='use greedy predict structure method when predicting')
	parser.add_argument('-ckys','--cky_predict_structure', action='store_true',
						help='use cky predict structure method when predicting')
	parser.add_argument('-ckyes','--cky_predict_edu_and_structure', action='store_false',
						help='use cky predict edu and structure jointly when predicting')
	
	#file paths
	parser.add_argument('-trp', '--train_path', default='raw_data/train',
						help='train corpus directory')
	parser.add_argument('-tep', '--test_path', default='raw_data/test',
						help='test corpus directory')
	#gpu					
	parser.add_argument('-g', '--gpu', action='store_false',
						help='whether to use gpu')
	
	return parser.parse_args()

args = process_commands()