"""
COMS W4701 Artificial Intelligence - Homework 0

In this assignment you will implement a few simple functions reviewing
basic Python operations and data structures.

@author: YOUR NAME (YOUR UNI)
"""


def manip_list(list1, list2):
    # YOUR CODE HERE
    # Print the last element of list1.
    print(list1[-1])

    # Remove the last element of list1.
    del list1[-1]

    # Change the second element of list2 to be identical to the first element of list1.
    list1[1] = list1[0]

    # Print a concatenation of list1 and list2 without modifying the two lists.
    print(list1, list2)

    # Return a single list consisting of list1 and list2 as its two elements.
    list = [list1, list2]
    return list


def manip_tuple(obj1, obj2):
    # YOUR CODE HERE
    # Create a tuple of the two object parameters.
    tuple1 = (obj1, obj2)

    # Attempt to modify the tuple by reassigning the first item -- Python should throw an exception upon execution.
    tuple1[0] = 100
    # There shows "Can't modify a tuple!"
    return None


def manip_set(list1, list2, obj):
    # YOUR CODE HERE
    # Create a set called set1 using list1.
    set_1 = set(list1)

    # Create a set called set2 using list2.
    set_2 = set(list2)

    # Add obj to set1.
    set_1.add(obj)

    # Test if obj is in set2 (Print True Or False)
    #print(obj in set_2)
    if obj not in set_2:

    # Print the difference of set1 and set2.
    print(set_1 - set_2)

    # Print the intersection of set1 and set2.
    print(set_1.intersection(set_2))

    # Remove obj from set1.
    set_1.remove(obj)

    return None


def manip_dict(tuple1, tuple2, obj):
    # YOUR CODE HERE
    # Create a dictionary such that elements of tuple1 serve as the keys for elements of tuple2.
    dict_1 = dict(zip(tuple1,tuple2))

    # Print the value of the dictionary mapped by obj.
    print(dict_1[obj])

    # Delete the dictionary pairing with the obj key.
    del dict_1[obj]

    # Print the length of the dictionary.
    print(len(dict_1))

    # Add a new pairing to the dictionary mapping from obj to the value 0.
    dict_1[obj] = 0

    # Return a list in which each element is a two-tuple of the dictionary's key-value pairings.
    list3 = []
    for item in dict_1.items():
        list3.append(item)

    return list3

if __name__ == "__main__":
    #Test case
    print(manip_list(["artificial", "intelligence", "rocks"], [4701, "is", "fun"]))

    try: manip_tuple("oh", "no")
    except TypeError: print("Can't modify a tuple!")

    manip_set(["sets", "have", "no", "duplicates"], ["sets", "operations", "are", "useful"], "yeah!")

    print(manip_dict(("list", "tuple", "set"), ("ordered, mutable", "ordered, immutable", "non-ordered, mutable"), "tuple"))