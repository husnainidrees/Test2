from collections import deque







def wl_bfs(start, end, list_word):

    # hum ny store is liye krwaya ha k unique word nikal sako

    cur_word = list(list_word)
 
    # agar idher path ni dy gy to hum age nikal nai sakty pop ni kr sakty

    queue = deque([(start,1)])  # (list k word, depth)
    print(queue)
    
    while queue:


        # hum ny pop is liye use nai kiya h wo jis index pr hota ha waha sy remove krta ha
        # jo left ha yeh us k next element k remove krta ha 
        word, path = queue.popleft()
        
        if word == end:
            return path
        
        print(word)
        # print(end)

        alpha_ord='abcdefghijklmnopqrstuvwxyz'

        for i in range(len(word)):
        
            for j in alpha_ord:
                next_word = word[:i] + j + word[i+1:]

                print(next_word)
                
                if next_word in cur_word:
                    # jo word hamry pas arhy ha us k append krna queue mein

                    queue.append((next_word, path + 1))
                    print(cur_word)
                    

    return 0  




strtWord = "bat"
endcha = "cat"
words = ["saw","dat", "rat", "dat", "hat", "cat"]

result=wl_bfs(strtWord, endcha, words)

print(result)








# hum ny is mein use krna ha

# def wl_bfs(start,end,list_word):

#     # jo start mean being word ha wo ajye ga

#     visited=set(start)

#     queue=deque([(start,1)])
#     # while is liye use kiya ha taky remove kr saky
#     while queue:
#         # queu mein sy next word k remove krta ha
#         # jo current word hota ha wo aye ga
#         c_word,path=queue.popleft()
        


#         alpha_data= 'abcdefghijklmnopqrst'
        
#         if c_word == endWord:
#             return path
#         for i in range (len(c_word)):
#             for j in  alpha_data:
#                 next_word = c_word[:i]  + j + c_word[i+1:]
#                 if next_word in visited:
#                     queue.append((next_word, path + 1))
#                     c_word.remove(next_word)  # Prevent cycles

#     return 0
