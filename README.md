# discourse-parser
demo mode instruction:  
python parser.py -demo -i <input_file> -o <output_file>


input_file  
Chinese raw text (utf-8) (simplified)


output_file  
json format:  
  {  
    'EDUs':[<edu1>, <edu2>...<edun>]  
    'tree':{'args':[<subtree>], 'sense':<sense>, 'center':<center>}  
    'relations':[{'arg1':<arg1>,'arg2':<arg2>,'sense':<sense>,'center':<center>},{...},{...}]  
  }
