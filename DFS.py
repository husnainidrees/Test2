




# jo dfs ha wo depth tak jata ha start sy end tak


def wl_dfs(start, end, list_word):


    # yeh code hum ny jo list banai ha us k store krta ha for unique value
    # cur_word=list(list_word) -> yeh word repeat kr deta ha set ni krta repeat
    # cur_word = set(list_word)
    
    cur_word = list(list_word)
    # print(cur_word)
 

    # stack = ([(start, [start])])  # (current_word, depth)
    stack = ([(start, [start])])  # (current_word, depth)
    print(stack)
    

    while stack:

        # yeh jo ha stack sy ik pop krtaha 

        word, path = stack.pop()
        
        if word == end:
            return path
        # print(word)
        print("For path")
        print(path)
        
        # 26 combination tak bany gy yeh alphabet k lihaz sy 

       

        alpha_ord='abcdefghijklmnopqr'
        for i in range(len(word)):

            for char in alpha_ord:


                # slicing is liye k ha k start sy i k range tak value chaly hamari 

                pred_word = word[:i] + char + word[i+1:]
                print("Next word kese arha ha")
                print(pred_word)
                
                
                if pred_word in cur_word:
                    stack.append((pred_word, path + [pred_word]))
                    print(path)
                    cur_word.remove(pred_word)
                    print(cur_word)
                    print(stack)
                    # print(pred_word) 

    return 0  

# strtWord = "hit"
# endcha = "cog"
# words = ["hot","wow", "dog", "lot", "log", "cog"]

# result=wl_dfs(strtWord, endcha, words)
# print(result)  

strtWord = "bat"
endcha = "cat"
words = ["saw","dat", "rat", "dat", "hat", "cat"]

result=wl_dfs(strtWord, endcha, words)

print(result)
