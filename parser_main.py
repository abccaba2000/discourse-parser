'''main python file'''

# the Reporter class is use to help log, statistics, debug, measure process time, an instance reporter is initialized when first import
import report
#to get arguments from shell 
#execute and stored the command arguments in variable args when first import
import args
#report.reporter and args.args are the two global objects shared by all files that import report.py and args.py

#for preprocess: load corpus
import utility

import model_utility


	
def main():
	#get the corpus
	#to get a list of Corpus objects defined in corpus.py, each object corresponds to a paragraph
	if args.args.train:	
		#if '-tr' or '--train' is specified, we would get both a list for training and a list for test
		pass
	train_corpus_list = utility.get_corpus_list(args.args.train_path)
	test_corpus_list = utility.get_corpus_list(args.args.test_path)
	
	
	if args.args.train:
		model = model_utility.get_model_lstm_rvnn()
		model_utility.train(model, train_corpus_list)			
	
	if args.args.load_model != None:
		model = model_utility.get_model_lstm_rvnn(args.args.load_model)
	model_utility.test(model, test_corpus_list)
	
	return

if __name__ == "__main__":
	main()