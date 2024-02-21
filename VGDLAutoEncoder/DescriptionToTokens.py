
import csv
import math 

class TokenDict:
    def __init__(self, words = []) -> None:
        self.encoder = {} 
        self.decoder = {}
        self.next_token = 0 
        for word in words:
            if(not word in self.encoder):
                self.encoder[word] = self.next_token 
                self.decoder[self.next_token] = word 
                self.next_token += 1 
    
    def encode(self, word):
        if(not word in self.encoder):
            self.encoder[word] = self.next_token 
            self.decoder[self.next_token] = word 
            self.next_token += 1 
            
        return self.encoder[word]
    
    def decode(self, token):
        if(not token in self.decoder):
            return -1 
        else:
            return self.decoder[token]
            



def fileToWords(file_name):
    file = open(file_name)
    desc = file.readlines()
    words = []

    for line in desc: 

        # remove comment 
        comment_index = line.find("#")
        if(comment_index != -1):
            line = line[:comment_index]
        
        line_words = line.split(" ")
        
        space_builder = "" 
        for word in line_words:
            if(word == ""):
                space_builder += " "
            else:
                if(len(space_builder) > 0):
                    words.append(space_builder)
                    space_builder = ""
                if(word.endswith("\n")):
                    word = word.replace("\n", "")
                words.append(word)
        words.append("\n") 
    
    return words 



def re_create(words):
    for word in words:
        print(word + " ", end="")

def codes_to_string(codes, code_dict):
    words = []
    for code in codes:
        if(code < 0):
            words.append("NA")
        else:
            words.append(code_dict.decode(round(code)))
    return " ".join(words)
 
def get_codes(csv_file):
    codes = []
    token_codes = TokenDict()
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            words = fileToWords(row[1])
            code = [token_codes.encode(word) for word in words]
            codes.append(code)
    
    return codes, token_codes

if __name__ == "__main__":
    file = "examples/all_games_sp.csv" 
    codes, token_code = get_codes(file)

    print(codes_to_string(codes[4], token_code))
    print(token_code.next_token)
    length = [len(code) for code in codes]
    print(max(length))
    
    