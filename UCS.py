





def path_cost(path):
    total_cost = 0
    for  cost in path:
        total_cost += cost
        print(type(total_cost))

        # total_cost+=cost
        # yeh retun mein ha jo agar path same howa to alphabetically kry ga
    return total_cost


# jo dfs ha wo depth tak jata ha start sy end tak


def wl_ucs(start, end, list_word):


    # yeh code hum ny jo list banai ha us k store krta ha for unique value
    cur_word = set(list_word)
    # print(cur_word)
 

    # stack = ([(start, [start])])  # (word, path)
    # jo hamra word ha wohi hamara depth b ha
    queue = ([(start, [start])])  # (word, path)
    print(queue)
    
    
    while queue:

        # yeh jo ha stack sy ik pop krtaha 
        # value sorted ho jati ha 

        queue.sort(key=path_cost)
        # word,path, cost = min(queue )

        word ,path=queue.pop()

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
                    
                    r = queue.append((pred_word,path + [pred_word]))
                    
                    print(r)
                    
                    cur_word.remove(pred_word)
                    
                    # print(cur_word)
                    # print(queue)
                    # print(pred_word) 

    return 0  

strtWord = "hit"
endcha = "cog"
words = ["hot","wow", "dog", "lot", "log", "cog"]

result=wl_ucs(strtWord, endcha, words)
print(result)  









def path_cost(path):
    # Cost is calculated as the length of the path (number of transformations)
    total_cost = len(path) - 1  # Cost is one less than the number of words in the path
    return total_cost


def wl_ucs(start, end, list_word):
    cur_word = set(list_word)  # Unique words to explore
    
    # Queue holds tuples of (current word, path from start to current word)
    queue = [(start, [start])]  # (word, path)
    
    while queue:
        # Sort queue by the path cost (UCS behavior)
        queue.sort(key=lambda x: path_cost(x[1]))
        
        # Extract the word and its current path
        word, path = queue.pop(0)
        
        if word == end:
            return path, path_cost(path)  # Return path and its cost
        
        # Generate next words by changing each letter
        alpha_ord = 'abcdefghijklmnopqrstuvwxyz'
        for i in range(len(word)):
            for char in alpha_ord:
                pred_word = word[:i] + char + word[i+1:]
                
                if pred_word in cur_word:
                    queue.append((pred_word, path + [pred_word]))
                    cur_word.remove(pred_word)  # Mark as visited

    return [], 0  # Return empty path and zero cost if no path found


# Test the UCS-based Word Ladder
start_word = "hit"
end_word = "cog"
word_list = ["hot", "dot", "dog", "lot", "log", "cog"]

result_path, total_cost = wl_ucs(start_word, end_word, word_list)
print("Shortest Path:", result_path)
print("Total Cost (transformations):", total_cost)

