import React, { useState } from "react";

// Sample Python programs
const programs = [
  { 
    title: "Program 1 - Bubble Sort", 
    code: `
import matplotlib.pyplot as plt
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def generate_decreasing_list(n):
    return list(range(n, 0, -1))

sizes = [100, 200, 300, 400, 500]  
times = []

for size in sizes:
    arr = generate_decreasing_list(size)
    start_time = time.time()
    bubble_sort(arr)
    end_time = time.time()
    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"List size: {size}, Time taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title('Bubble Sort Time Complexity (Decreasing Order)')
plt.xlabel('List Size')
plt.ylabel('Time Taken (seconds)')
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://www.youtube.com/embed/xli_FI7CuzA"
  },
  { 
    title: "Program 2 - Linear Search", 
    code: `
import matplotlib.pyplot as plt
import time

def linearsearch(arr, key):
    for i in range(0, len(arr)):
        if arr[i] == key:
            print(f"Element {key} found at index {i+1}")
            return i
    print(f"Element {key} not found")
    return -1

def generate_decreasing_list(n):
    return list(range(n, 0, -1))

sizes = [100, 200, 300, 400, 500]
times = []

for size in sizes:
    arr = generate_decreasing_list(size)
    # Search for an element that will be at the end of the list to simulate worst-case
    key_to_find = 1
    start_time = time.time()
    linearsearch(arr, key_to_find)
    end_time = time.time()
    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"List Size: {size}, Time Taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title("Linear Search (Decreasing List order)")
plt.xlabel("List size")
plt.ylabel("Time taken")
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://youtu.be/246V51AWwZM?si=wJsGKZP1L5sYPCcL"
  },
  { 
    title: "Program 3 - Selection Sort", 
    code: `
import matplotlib.pyplot as plt
import time

def selectionsort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def generate_decreasing_list(n):
    return list(range(n, 0, -1))

sizes = [100, 200, 300, 400, 500]
times = []

for size in sizes:
    arr = generate_decreasing_list(size)
    start_time = time.time()
    selectionsort(arr)
    end_time = time.time()
    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"List size: {size}, Time taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title("Selection Sort (decreasing list)")
plt.xlabel("List size")
plt.ylabel("Time taken")
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://www.youtube.com/watch?v=g-PGLbMth_g"
  },
  {
    title: "Program 4 - Insertion Sort",
    code: `
import matplotlib.pyplot as plt
import time

def insertionsort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j = j - 1
        arr[j + 1] = key

def generate_decreasing_list(n):
    return list(range(n, 0, -1))

sizes = [100, 200, 300, 400, 500]
times = []

for size in sizes:
    arr = generate_decreasing_list(size)
    start_time = time.time()
    insertionsort(arr)
    end_time = time.time()
    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"List size: {size}, Time taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title("Insertion Sort Time Complexity (Decreasing Order)")
plt.xlabel("List size")
plt.ylabel("Time taken")
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://www.youtube.com/embed/JU7OW5j4s2U"
  },
  {
    title: "Program 5 - Binary Search",
    code: `
import matplotlib.pyplot as plt
import time

def binarysearch(arr, low, high, key):
    if high < low:
        print("Element not present in array")
        return
    
    mid = int(low + (high - low) / 2)
    
    if arr[mid] == key:
        print("Element present at index:", mid)
        return
    elif arr[mid] > key:
        return binarysearch(arr, low, mid - 1, key)
    else:
        return binarysearch(arr, mid + 1, high, key)

def generate_sorted_list(n):
    return list(range(1, n + 1))

sizes = [100, 200, 300, 400, 500]
times = []

for size in sizes:
    arr = generate_sorted_list(size)
    start_time = time.time()
    # Search for a value in the middle to represent an average case
    key_to_find = size // 2
    binarysearch(arr, 0, len(arr) - 1, key_to_find)
    end_time = time.time()
    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"List size: {size}, Time taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title("Binary Search Time Complexity (Sorted List)")
plt.xlabel("List size")
plt.ylabel("Time taken")
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://www.youtube.com/embed/f_tS95_V04Y"
  },
  {
    title: "Program 6 - Merge Sort",
    code: `
import matplotlib.pyplot as plt
import time

def mergesort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        b = arr[:mid]
        c = arr[mid:]
        mergesort(b)
        mergesort(c)
        i = j = k = 0
        while i < len(b) and j < len(c):
            if b[i] < c[j]:
                arr[k] = b[i]
                i += 1
            else:
                arr[k] = c[j]
                j += 1
            k += 1
        while i < len(b):
            arr[k] = b[i]
            i += 1
            k += 1
        while j < len(c):
            arr[k] = c[j]
            j += 1
            k += 1

def generate_decreasing_list(n):
    return list(range(n, 0, -1))

sizes = [100, 200, 300, 400, 500]
times = []

for size in sizes:
    arr = generate_decreasing_list(size)
    start_time = time.time()
    mergesort(arr)
    end_time = time.time()
    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"List size: {size}, Time taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title("Merge Sort Time Complexity (Decreasing Order)")
plt.xlabel("List size")
plt.ylabel("Time taken")
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://www.youtube.com/embed/bS1-uYv8-zE"
  },
  {
    title: "Program 7 - Quick Sort",
    code: `
import matplotlib.pyplot as plt
import time

def partition(arr, low, high):
    pivot=arr[high]
    i=low-1
    for j in range (low,high):
        if arr[j]<=pivot:
            i=i+1
            arr[j],arr[i]=arr[i],arr[j]
    arr[i+1],arr[high]=arr[high], arr[i+1]
    return i+1

def quicksort(arr, low, high):
    if low<high:
        p=partition(arr,low,high)
        quicksort(arr, low, p-1)
        quicksort(arr, p+1, high)

def generate_decreasing_list(n):
    return list (range(n,0,-1))

sizes=[100,200,300,400,500]
times=[]

for size in sizes:
    arr=generate_decreasing_list(size)
    start_time=time.time()
    quicksort(arr, 0, size-1)
    end_time=time.time()
    time_taken=end_time-start_time
    times.append(time_taken)
    print(f"List size: {size}, Time taken: {time_taken:.6f} seconds")

plt.figure(figsize=(10,6))
plt.plot(sizes,times,marker='o',linestyle='-',color='b')
plt.title("Quick sort time complexity (Decreasing Order)")
plt.xlabel("List size")
plt.ylabel("Time taken")
plt.xticks(sizes)
plt.grid(True)
plt.show()`,
    videoUrl: "https://www.youtube.com/embed/S5f-v-c5wK8"
  },
  {
    title: "Program 8 - Fibonacci",
    code: `def fibonacci(n):
    arr=[0,1]
    for i in range (2,n+1):
        arr.append(arr[i-1] + arr[i-2])
    return arr
    
n=10
print(fibonacci(n))`,
    videoUrl: "https://www.youtube.com/embed/dxyYP3BSdcQ"
  },
  {
    title: "Program 9 - Singly Linked List",
    code: `class Node:
    def __init__(self, data):
        self.data = data
        self.Next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def InsertAtBegin(self, data):
        new = Node(data)
        new.Next = self.head
        self.head = new

    def InsertAtEnd(self, data):
        new = Node(data)
        if self.head is None:
            self.head = new
            return
        cur = self.head
        while cur.Next is not None:
            cur = cur.Next
        cur.Next = new

    def DelAtBegin(self):
        if self.head is None:
            print("List is empty, nothing to delete.")
            return
        print("Deleted item =", self.head.data)
        self.head = self.head.Next

    def delAtEnd(self):
        if self.head is None:
            print("List is empty, nothing to delete.")
            return
        elif self.head.Next is None:
            print("Deleted item =", self.head.data)
            self.head = None
            return
        cur = self.head
        while cur.Next.Next is not None:
            cur = cur.Next
        print("Deleted item =", cur.Next.data)
        cur.Next = None

    def Search(self, key):
        cur = self.head
        while cur is not None:
            if cur.data == key:
                print(f"{key} found in the list.")
                return
            cur = cur.Next
        print(f"{key} not found in the list.")

    def display(self):
        if self.head is None:
            print("List is empty")
            return
        cur = self.head
        while cur is not None:
            print(cur.data, end=" -> ")
            cur = cur.Next
        print("None")

# Example Usage
ll = LinkedList()
ll.InsertAtBegin(10)
ll.InsertAtEnd(20)
ll.InsertAtEnd(30)
ll.display()

ll.DelAtBegin()  # Deletes 10
ll.display()

ll.delAtEnd()  # Deletes 30
ll.display()

ll.Search(20)  # Found
ll.Search(50)  # Not Found`,
    videoUrl: "https://www.youtube.com/embed/NCu_vR1mJmE"
  },
  {
    title: "Program 10 - Doubly Linked List",
    code: `class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def insertAtBegin(self, data):
        new = Node(data)
        if self.head is None:
            self.head = new
            return
        new.next = self.head
        self.head.prev = new
        self.head = new

    def insertAtEnd(self, data):
        new = Node(data)
        if self.head is None:
            self.head = new
            return
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = new
        new.prev = cur 

    def display(self):
        if self.head is None:
            print("The list is empty")
            return
        cur = self.head
        while cur is not None:
            print(cur.data, end=" <-> ")
            cur = cur.next
        print("None")

    def deleteAtEnd(self):
        if self.head is None:
            print("List is empty")
            return
        elif self.head.next is None:
            print("Deleted:", self.head.data)
            self.head = None
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            print("Deleted:", cur.data)
            cur.prev.next = None

    def deleteAtBegin(self):
        if self.head is None:
            print("List is empty")
            return
        elif self.head.next is None:
            print("Deleted:", self.head.data)
            self.head = None
        else:
            print("Deleted:", self.head.data)
            self.head = self.head.next
            self.head.prev = None

    def search(self, key):
        cur = self.head
        while cur is not None:
            if cur.data == key:
                print(f"{key} found in the list.")
                return
            cur = cur.next
        print(f"{key} not found in the list.")

# Example Usage
dll = DoublyLinkedList()
dll.insertAtBegin(10)
dll.insertAtEnd(20)
dll.insertAtEnd(30)
dll.display()

dll.search(20)  # Found
dll.search(40)  # Not Found`,
    videoUrl: "https://www.youtube.com/embed/MOoYxzYFAtU"
  },
  {
    title: "Program 11 - Circular Doubly Linked List",
    code: `class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class CircularDoublyLinkedList:
    def __init__(self): 
        self.head = None

    def insert_at_begin(self, data):
        new = Node(data)
        if not self.head:
            self.head = new
            self.head.next = self.head
            self.head.prev = self.head
        else:
            last = self.head.prev
            new.next = self.head
            new.prev = last
            self.head.prev = new
            last.next = new
            self.head = new

    def insert_at_end(self, data):
        new = Node(data)
        if not self.head:
            self.head = new
            self.head.next = self.head
            self.head.prev = self.head
        else:
            last = self.head.prev
            last.next = new
            new.prev = last
            new.next = self.head
            self.head.prev = new

    def delete_at_begin(self):
        if not self.head:
            return
        if self.head.next == self.head:
            self.head = None
        else:
            last = self.head.prev
            self.head = self.head.next
            self.head.prev = last
            last.next = self.head

    def delete_at_end(self):
        if not self.head:
            return
        if self.head.next == self.head:
            self.head = None
        else:
            last = self.head.prev
            second_last = last.prev
            second_last.next = self.head
            self.head.prev = second_last

    def search(self, key):
        if not self.head:
            print(f"{key} not found in the list.")
            return
        temp = self.head
        while True:
            if temp.data == key:
                print(f"{key} found in the list.")
                return
            temp = temp.next
            if temp == self.head:
                break
        print(f"{key} not found in the list.")

    def display(self):
        if not self.head:
            print("List is empty")
            return
        temp = self.head
        while True:
            print(temp.data, end=" <-> ")
            temp = temp.next
            if temp == self.head:
                break
        print()

// Example usage
cdll = CircularDoublyLinkedList()
cdll.insert_at_begin(1)
cdll.insert_at_end(2)
cdll.insert_at_end(3)
cdll.display()

cdll.search(2)
cdll.search(4)

cdll.delete_at_begin()
cdll.display()

cdll.delete_at_end()
cdll.display()`,
    videoUrl: "https://www.youtube.com/embed/DrATf9rtfoE"
  },
  {
    title: "Program 12 - Stack Data Structure",
    code: `class Stack:
    def __init__(self):
        self.items=[]
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if len(self.items) == 0:
            print("list is empty")
        else:
            self.items.pop()
        
    def peek(self):
        if len(self.items)==0:
            print("list is empty")
        else:
            print("Stack: ", self.items[-1])
    
    def size(self):
        print("Size of stack: ", len(self.items))

    def display(self):
        if len(self.items)==0:
            print("Empty List")
        else:
            print("Items: ", self.items)

stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
stack.size()
stack.peek()
stack.pop()
stack.display()`,
    videoUrl: "https://www.youtube.com/embed/i4YZl2cQmZ8"
  },
  {
    title: "Program 13 - Bracket Matching",
    code: `def bracketMatch(myStr):
    stack = []
    bracket_map = {')': '(', '}': '{', ']': '['}

    for char in myStr:
        if char in bracket_map.values():
            stack.append(char)
        elif char in bracket_map:
            if not stack or stack[-1] != bracket_map[char]:
                return "Unbalanced"
            stack.pop()

    return "Balanced" if not stack else "Unbalanced"

# Test cases
print(bracketMatch("{[]{()}}"))  
print(bracketMatch("[{}{})(]"))  
print(bracketMatch("((()))"))     
print(bracketMatch("{[()]}"))     
print(bracketMatch("{[(])}"))     `,
    videoUrl: "https://www.youtube.com/embed/hIZ9RK2izOU"
  },
  {
    title: "Program 14 - Factorial",
    code: `def factorial(n):
  if (n < 0 or int(n) != n): 
      return "Not defined"  
  if (n == 1 or n == 0):  
      return 1
  else:
      return n * factorial(n - 1)  

f = int(input('Enter the number: '))
print("factorial of a given number = ",factorial(f))`,
    videoUrl: "https://www.youtube.com/embed/nh-ZBm91Wus"
  },
  {
    title: "Program 15 - Fibonacci Recursive",
    code: `def fib(n):
  if n < 0 or int(n) != n:
      return "Not defined"
  elif n == 0 or n == 1 :
      return n
  else:
      return fib(n-1) + fib(n-2)

n = int(input('Enter the number: '))
print("Fibonacci series :")
for i in range(0, n):
  print(fib(i))`,
    videoUrl: "https://www.youtube.com/embed/dxyYP3BSdcQ"
  },
  {
    title: "Program 16 - Towers of Hanoi",
    code: `def Towers(disks, source, auxialiary, target):
    if (disks==1):
        print("Move disk1 from rod {} to rod {}".format(source, target))
        return
    Towers(disks -1, source, target, auxialiary)
    print("Move disk{} from rod{} to rod{}".format(disks,source, target))
    Towers(disks -1, auxialiary, source, target)

disks=int (input("Enter the no disks"))
Towers(disks, 'A','B','C')`,
    videoUrl: "https://www.youtube.com/embed/S4A-kQ5l0m0"
  },
  {
    title: "Program 17 - Queue",
    code: `from collections import deque

class Queue:
    def __init__(self):
        self.head = deque()

    def enqueue(self, item):
        self.head.append(item)
        print(f"Enqueued: ", item)

    def dequeue(self):
        if len(self.head) == 0:        
            print("Queue is empty. Cannot dequeue.")
        else:
            removed_item = self.head.popleft()
            print(f"Dequeued: ", removed_item)

    def peek(self):
        if len(self.head) == 0:
            print("Queue is empty. No front item.")
        else:
            print(f"Front item: ", self.head[0])

    def display(self):
        print(f"Queue contents: ",list(self.head))

# Demonstration
queue = Queue()
queue.enqueue(103)
queue.enqueue(202)
queue.enqueue(30)
queue.display()
queue.dequeue()
queue.display()
queue.peek()
queue.dequeue()
queue.dequeue()
queue.dequeue()`,
    videoUrl: "https://www.youtube.com/embed/agQQyKd0HBQ"
  },
  {
    title: "Program 18 - Priority Queue",
    code: `import heapq

class PriorityQueue:
    def __init__(self):
        self.head = []

    def enqueue(self, item, priority):
        heapq.heappush(self.head, (priority, item))
        print("Enqueued: ",item," with priority ",priority)

    def dequeue(self, priority):
        if len(self.head) == 0:            
            print("Priority Queue is empty. Cannot dequeue.")
        else:
            priority, item = heapq.heappop(self.head)
            print(f"Dequeued: {item} with priority {priority}")
    
    def peek(self):
        if len(self.head) == 0:            
            print("Priority Queue is empty. No items to peek.")
        else:            
            priority, item = self.head[0]
            print(f"Highest priority item: {item} with priority {priority}")
    
    def display(self):
        if len(self.head) == 0:            
            print("Priority Queue is empty. No items to peek.")
        else:        
            print("Priority Queue contents:")
            for priority, item in sorted(self.head):
              print(item, "Priority: ",priority)

// Demonstration
pq = PriorityQueue()
pq.enqueue("Task A", 3)
pq.enqueue("Task B", 1)
pq.enqueue("Task C", 2)
pq.display()

pq.dequeue()
pq.display()

pq.peek()
pq.dequeue()
pq.dequeue()
pq.dequeue()`,
    videoUrl: "https://www.youtube.com/embed/-D6JYBrVrhs"
  },
  {
    title: "Program 19 - Binary Search Tree (Array-based)",
    code: `class BinarySearchhead:
    def __init__(self):
        self.head = []

    def insert(self, value):
        if not self.head:
            self.head.append(value)
        else:
            index = 0
            while True:
                # Expand the list to fit the required index
                if index >= len(self.head):
                    self.head.extend([None] * (index - len(self.head) + 1))
                
                if self.head[index] is None:
                    self.head[index] = value
                    break
                
                if value < self.head[index]:
                    index = 2 * index + 1  # Left child
                else:
                    index = 2 * index + 2  # Right child
    
    def search(self, value):
        index = 0
        while index < len(self.head):
            if self.head[index] is None:
                print(f"{value} not found in the head.")
                return False
            if self.head[index] == value:
                print(f"{value} found in the head at index {index}.")
                return True
            elif value < self.head[index]:
                index = 2 * index + 1
            else:
                index = 2 * index + 2
        print(f"{value} not found in the head.")
        return False

    def inorder_traversal(self, index=0):
        if index < len(self.head) and self.head[index] is not None:
            self.inorder_traversal(2 * index + 1)
            print(self.head[index], end=" ")
            self.inorder_traversal(2 * index + 2)

    def display(self):
        print(f"head structure: {self.head}")

# Demonstration
bst = BinarySearchhead()
bst.insert(50)
bst.insert(90)
bst.insert(70)
bst.insert(20)
bst.insert(40)
bst.insert(60)
bst.insert(80)
print("In-order Traversal:", end=" ")
bst.inorder_traversal()
print()
bst.search(70)
bst.search(90)
bst.display()`,
    videoUrl: "https://www.youtube.com/embed/j_Nf51S2IqA"
  },
  {
    title: "Program 20 - Breadth-First Search (BFS)",
    code: `from collections import deque
def bfs(graph, start):
    visited = set()  
    queue = deque([start]) 
    while queue:
        node = queue.popleft()  
        if node not in visited:
            print(node)  
            visited.add(node)  

            for i in graph[node]:
                if i not in visited:
                    queue.append(i)

// Example graph represented as an adjacency list
graph = {
    'A': ['B', 'C','D'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("BFS Traversal starting from node 'A':")
bfs(graph,'A')`,
    videoUrl: "https://www.youtube.com/embed/agQQyKd0HBQ"
  },
  {
    title: "Program 21 - DFS",
    code: `def dfs(node, vis):
    if node not in vis:
        print(node, end=" ")  
        vis.add(node)  
        for i in graph.get(node, []):  
            dfs(i, vis)

graph = {
    'A': ['B', 'C'],
    'B': ['E', 'F'],
    'C': ['G'],
    'E': ['D', 'H'],
    'F': [],
    'G': [],
    'D': ['A'],
    'H': []
}
print("Recursive DFS Traversal:")
dfs('A', set())`,
    videoUrl: "https://www.youtube.com/embed/iaBEKo5sM7w"
  },
  {
    title: "Program 22 - Hash Table",
    code: `class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size  
    
    def insert(self, key, value):
        index = hash(key) % self.size
        while self.table[index] is not None:
            index = (index + 1) % self.size  
        self.table[index] = (key, value)
    
    def search(self, key):
        index = hash(key) % self.size
        while self.table[index] is not None:
            if self.table[index][0] == key:
                return self.table[index][1]  
            index = (index + 1) % self.size
        return None 
    
    def delete(self, key):
        index = hash(key) % self.size
        while self.table[index] is not None:
            if self.table[index][0] == key:
                self.table[index] = None  
                return True
            index = (index + 1) % self.size
        return False  
    
    def display(self):
        print("Hash Table Contents:")
        for index, entry in enumerate(self.table):
            if entry is not None:
                print(f"Index {index}: {entry[0]} --------> {entry[1]}")
            else:
                print(f"Index {index}: Empty")

a = HashTable(5)
a.insert("apple", 10)
a.insert("banana", 20)
a.insert("cherry", 30)
a.display()
print("Value for apple:", a.search('apple'))
print("Value for banana", a.search('banana'))
print("Value for cherry:", a.search('cherry'))
print("Value for orange", a.search('orange'))  
a.delete("banana")
a.delete("cherry")
print("\\nHash table after deletion of 'banana':")
a.display()`,
    videoUrl: "https://www.youtube.com/embed/VeYKEMY2F9k"
  }
];

function ProgramViewer() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const prevProgram = () => setCurrentIndex((prev) => (prev > 0 ? prev - 1 : prev));
  const nextProgram = () => setCurrentIndex((prev) => (prev < programs.length - 1 ? prev + 1 : prev));

  const currentCode = programs[currentIndex].code;
  const codeLines = currentCode.split('\n');
  const currentVideoUrl = programs[currentIndex].videoUrl;

  // A simple function to apply syntax highlighting with inline styles
  const highlightSyntax = (line) => {
    // Regex for keywords, functions, comments, numbers, and strings
    const patterns = {
      keyword: /\b(import|from|def|for|if|else|return|in|range|while|and|class|elif)\b/g,
      function: /\b([a-zA-Z_]\w*)\s*\(/g,
      comment: /#.*/g,
      string: /(['"])(?:(?!\1).)*\1/g,
      number: /\b\d+\b/g,
    };

    const colors = {
      keyword: '#9333ea',
      function: '#eab308',
      comment: '#a1a1aa',
      string: '#16a34a',
      number: '#3b82f6',
      default: '#374151',
    };

    let highlightedLine = [line];
    let nextHighlight = [];

    // Order of precedence: comments, strings, keywords, numbers, functions
    const applyHighlight = (key) => {
      nextHighlight = [];
      highlightedLine.forEach(segment => {
        if (typeof segment !== 'string') {
          nextHighlight.push(segment);
          return;
        }

        let lastIndex = 0;
        let match;
        const regex = patterns[key];

        while ((match = regex.exec(segment)) !== null) {
          if (match.index > lastIndex) {
            nextHighlight.push(segment.substring(lastIndex, match.index));
          }
          nextHighlight.push(<span key={`${key}-${match.index}`} style={{ color: colors[key] }}>{match[0]}</span>);
          lastIndex = regex.lastIndex;
        }
        if (lastIndex < segment.length) {
          nextHighlight.push(segment.substring(lastIndex));
        }
      });
      highlightedLine = nextHighlight;
    };

    applyHighlight('comment');
    applyHighlight('string');
    applyHighlight('keyword');
    applyHighlight('number');
    applyHighlight('function');

    return highlightedLine.map((segment, index) => (
      <React.Fragment key={index}>{segment}</React.Fragment>
    ));
  };


  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: '#e5e7eb', fontFamily: 'sans-serif', color: '#374151' }}>
      
      <header style={{
        backgroundColor: 'black',
        color: 'white',
        padding: '0.20rem 1rem', // Reduced vertical padding for less height
        textAlign: 'center',
        minHeight: '10px', // Reduced minimum height
        position: 'sticky', // Make header sticky
        top: 0, // Stick to top
        zIndex: 1000, // Ensure it stays above other content
      }}>
        <h1 style={{ fontSize: '1 rem', fontWeight: 'normal' }}>COMPUTER SCIENCE AND ENGINEERING</h1>
        <h2 style={{ fontSize: '0.700 rem', fontWeight: 'normal' }}>Data Structures with Python Lab Manual</h2>
      </header>

      <div style={{ display: 'flex', flex: '1' }}>
        <aside style={{ width: '200px', backgroundColor: '#9ca3af', color: 'black', padding: '0', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          <div style={{ padding: '0.75rem 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#d1d5db' }}>
            <h1 style={{ fontSize: '1rem', fontWeight: 'bold' }}>Lab Programs</h1>
          </div>
          {programs.map((program, index) => (
            <div
              key={index}
              style={{
                padding: '0.75rem 1rem',
                cursor: 'pointer',
                backgroundColor: index === currentIndex ? '#6b7280' : '#d1d5db',
                fontWeight: index === currentIndex ? '600' : 'normal',
                color: 'black',
                borderTop: '1px solid #d1d5db'
              }}
              onClick={() => setCurrentIndex(index)}
            >
              {program.title}
            </div>
          ))}
        </aside>

        <main style={{ flex: '1', padding: '2rem', backgroundColor: '#ffffff' }}>
          <div style={{ backgroundColor: '#ffffff', padding: '1rem' }}>
            <header style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  style={{
                    backgroundColor: '#d1d5db',
                    padding: '0.5rem 1rem',
                    fontSize: '0.875rem',
                    cursor: currentIndex === 0 ? 'not-allowed' : 'pointer',
                    border: '1px solid #9ca3af',
                    opacity: currentIndex === 0 ? 0.5 : 1,
                  }}
                  onClick={prevProgram}
                  disabled={currentIndex === 0}
                >
                  Prev
                </button>
                <button
                  style={{
                    backgroundColor: '#d1d5db',
                    padding: '0.5rem 1rem',
                    fontSize: '0.875rem',
                    cursor: currentIndex === programs.length - 1 ? 'not-allowed' : 'pointer',
                    border: '1px solid #9ca3af',
                    opacity: currentIndex === programs.length - 1 ? 0.5 : 1,
                  }}
                  onClick={nextProgram}
                  disabled={currentIndex === programs.length - 1}
                >
                  Next
                </button>
              </div>
            </header>

            <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>{programs[currentIndex].title}</h2>
            {currentIndex === 0 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Python program to use and demonstrate basic data structures</p>
            )}
            {currentIndex === 1 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement Linear Search, compute space and time complexities, and plot a graph using asymptotic notations.</p>
            )}
            {currentIndex === 2 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement Selection Sort, compute space and time complexities, and plot a graph using asymptotic notations.</p>
            )}
            {currentIndex === 3 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement Insertion Sort, compute space and time complexities, and plot a graph using asymptotic notations.</p>
            )}
            {currentIndex === 4 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement Binary Search, compute space and time complexities, and plot a graph using asymptotic notations.</p>
            )}
            {currentIndex === 5 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement Merge Sort, compute space and time complexities, and plot a graph using asymptotic notations.</p>
            )}
            {currentIndex === 6 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement Quick Sort, compute space and time complexities, and plot a graph using asymptotic notations.</p>
            )}
            {currentIndex === 7 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement the Fibonacci sequence using an iterative approach.</p>
            )}
            {currentIndex === 8 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Singly Linked List with various operations like insertion, deletion, searching, and displaying elements.</p>
            )}
            {currentIndex === 9 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Doubly Linked List with various operations like insertion, deletion, searching, and displaying elements.</p>
            )}
            {currentIndex === 10 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Circular Doubly Linked List with various operations like insertion, deletion, searching, and displaying elements.</p>
            )}
            {currentIndex === 11 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Stack Data Structure with push, pop, peek, size, and display operations.</p>
            )}
            {currentIndex === 12 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a program that checks for balanced brackets in an expression using a stack.</p>
            )}
            {currentIndex === 13 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement the factorial function using a recursive approach.</p>
            )}
            {currentIndex === 14 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement the Fibonacci sequence using a recursive approach.</p>
            )}
            {currentIndex === 15 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement the Towers of Hanoi algorithm using recursion to solve the classic puzzle.</p>
            )}
            {currentIndex === 16 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Queue data structure using `collections.deque` with enqueue, dequeue, peek, and display operations.</p>
            )}
            {currentIndex === 17 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Priority Queue using the `heapq` module to handle elements based on priority.</p>
            )}
            {currentIndex === 18 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Binary Search Tree using an array-based representation, demonstrating insertion, searching, and in-order traversal.</p>
            )}
            {currentIndex === 19 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement the Breadth-First Search (BFS) algorithm for graph traversal.</p>
            )}
            {currentIndex === 20 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement the Depth-First Search (DFS) algorithm for graph traversal.</p>
            )}
            {currentIndex === 21 && (
              <p style={{ color: '#4b5563', marginBottom: '0.5rem' }}>Implement a Hash Table with collision handling using linear probing, and perform insertion, search, and deletion operations.</p>
            )}
            <pre style={{
                padding: '1rem',
                backgroundColor: '#e5e7eb',
                color: '#374151',
                whiteSpace: 'pre-wrap',
                fontSize: '12px',
                fontFamily: 'monospace'
            }}>
              <div style={{ display: 'flex' }}>
                <div style={{ color: '#a1a1aa', paddingRight: '1rem', textAlign: 'right', userSelect: 'none' }}>
                  {codeLines.map((_, index) => (
                    <div key={index} style={{ fontSize: '12px' }}>{index + 1}</div>
                  ))}
                </div>
                <div>
                  {codeLines.map((line, index) => (
                    <div key={index} style={{ fontSize: '12px' }}>{highlightSyntax(line)}</div>
                  ))}
                </div>
              </div>
            </pre>
            {/* Conditional output for Program 1 */}
            {currentIndex === 0 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>List size: 100, Time taken: 0.001040 seconds</p>
<p>List size: 200, Time taken: 0.003370 seconds</p>
<p>List size: 300, Time taken: 0.006828 seconds</p>
<p>List size: 400, Time taken: 0.013537 seconds</p>
<p>List size: 500, Time taken: 0.021694 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 2 */}
            {currentIndex === 1 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>Element 1 is found at index 100</p>
<p>List Size: 100, Time Taken: 0.000000 seconds</p>
<p>Element 1 is found at index 200</p>
<p>List Size: 200, Time Taken: 0.000000 seconds</p>
<p>Element 1 is found at index 300</p>
<p>List Size: 300, Time Taken: 0.000000 seconds</p>
<p>Element 1 is found at index 400</p>
<p>List Size: 400, Time Taken: 0.000000 seconds</p>
<p>Element 1 is found at index 500</p>
<p>List Size: 500, Time Taken: 0.000000 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 3 */}
            {currentIndex === 2 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>List size: 100, Time taken: 0.000780 seconds</p>
<p>List size: 200, Time taken: 0.003290 seconds</p>
<p>List size: 300, Time taken: 0.007693 seconds</p>
<p>List size: 400, Time taken: 0.012586 seconds</p>
<p>List size: 500, Time taken: 0.022944 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 4 */}
            {currentIndex === 3 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>List size: 100, Time taken: 0.000987 seconds</p>
<p>List size: 200, Time taken: 0.003975 seconds</p>
<p>List size: 300, Time taken: 0.008981 seconds</p>
<p>List size: 400, Time taken: 0.016023 seconds</p>
<p>List size: 500, Time taken: 0.025001 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 5 */}
            {currentIndex === 4 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>Element present at index: 49</p>
<p>List size: 100, Time taken: 0.000000 seconds</p>
<p>Element present at index: 99</p>
<p>List size: 200, Time taken: 0.000000 seconds</p>
<p>Element present at index: 149</p>
<p>List size: 300, Time taken: 0.000000 seconds</p>
<p>Element present at index: 199</p>
<p>List size: 400, Time taken: 0.000000 seconds</p>
<p>Element present at index: 249</p>
<p>List size: 500, Time taken: 0.000000 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 6 */}
            {currentIndex === 5 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>List size: 100, Time taken: 0.000078 seconds</p>
<p>List size: 200, Time taken: 0.000201 seconds</p>
<p>List size: 300, Time taken: 0.000329 seconds</p>
<p>List size: 500, Time taken: 0.000632 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 7 */}
            {currentIndex === 6 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
<p>List size: 100, Time taken: 0.002578 seconds</p>
<p>List size: 200, Time taken: 0.010345 seconds</p>
<p>List size: 300, Time taken: 0.023211 seconds</p>
<p>List size: 400, Time taken: 0.041123 seconds</p>
<p>List size: 500, Time taken: 0.064102 seconds</p>
                </pre>
              </div>
            )}
            {/* Conditional output for Program 8 */}
            {currentIndex === 7 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
                </pre>
              </div>
            )}
            {/* Conditional output for Program 9 */}
            {currentIndex === 8 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
10 &gt; 20 &gt; 30 &gt; None
Deleted item = 10
20 &gt; 30 &gt; None
Deleted item = 30
20 &gt; None
20 found in the list.
50 not found in the list.
                </pre>
              </div>
            )}
            {/* Conditional output for Program 10 */}
            {currentIndex === 9 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
10 &lt;&gt; 20 &lt;&gt; 30 &lt;&gt; None
20 found in the list.
40 not found in the list.
                </pre>
              </div>
            )}
            {/* Conditional output for Program 11 */}
            {currentIndex === 10 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
1 &lt;&gt; 2 &lt;&gt; 3 &lt;&gt; 
2 found in the list.
4 not found in the list.
2 &lt;&gt; 3 &lt;&gt; 
2 &lt;&gt; 
                </pre>
              </div>
            )}
            {/* Conditional output for Program 12 */}
            {currentIndex === 11 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Size of stack:  3
Stack:  3
Items:  [1, 2]
                </pre>
              </div>
            )}
            {/* Conditional output for Program 13 */}
            {currentIndex === 12 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Balanced
Unbalanced
Balanced
Balanced
Unbalanced
                </pre>
              </div>
            )}
            {/* Conditional output for Program 14 */}
            {currentIndex === 13 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Enter the number: 5
factorial of a given number =  120
                </pre>
              </div>
            )}
            {/* Conditional output for Program 15 */}
            {currentIndex === 14 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Enter the number: 10
Fibonacci series :
0
1
1
2
3
5
8
13
21
34
                </pre>
              </div>
            )}
            {/* Conditional output for Program 16 */}
            {currentIndex === 15 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Enter the no disks3
Move disk1 from rod A to rod C
Move disk2 from rod A to rod B
Move disk1 from rod C to rod B
Move disk3 from rodA to rodC
Move disk1 from rod B to rod A
Move disk2 from rod B to rod C
Move disk1 from rod A to rod C
                </pre>
              </div>
            )}
            {/* Conditional output for Program 17 */}
            {currentIndex === 16 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Enqueued:  103
Enqueued:  202
Enqueued:  30
Queue contents:  [103, 202, 30]
Dequeued:  103
Queue contents:  [202, 30]
Front item:  202
Dequeued:  202
Dequeued:  30
Queue is empty. Cannot dequeue.
                </pre>
              </div>
            )}
            {/* Conditional output for Program 18 */}
            {currentIndex === 17 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Enqueued:  Task A with priority  3
Enqueued:  Task B with priority  1
Enqueued:  Task C with priority  2
Priority Queue contents:
Task B Priority:  1
Task C Priority:  2
Task A Priority:  3
Dequeued: Task B with priority 1
Priority Queue contents:
Task C Priority:  2
Task A Priority:  3
Highest priority item: Task C with priority 2
Dequeued: Task C with priority 2
Dequeued: Task A with priority 3
Priority Queue is empty. Cannot dequeue.
                </pre>
              </div>
            )}
            {/* Conditional output for Program 19 */}
            {currentIndex === 18 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
In-order Traversal: 20 40 50 60 70 80 90 
70 found in the head at index 5.
90 found in the head at index 2.
head structure: [50, 20, 90, None, 40, 70, None, None, None, None, None, 60, 80]
                </pre>
              </div>
            )}
            {/* Conditional output for Program 20 */}
            {currentIndex === 19 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
BFS Traversal starting from node 'A':
A
B
C
D
E
F
                </pre>
              </div>
            )}
            {/* Conditional output for Program 21 */}
            {currentIndex === 20 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Recursive DFS Traversal:
A B E D H F C G 
                </pre>
              </div>
            )}
            {/* Conditional output for Program 22 */}
            {currentIndex === 21 && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e5e7eb', color: '#374151', border: '1px solid #d1d5db' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Output</h3>
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}>
Hash Table Contents:
Index 0: Empty
Index 1: ('apple', 10)
Index 2: ('banana', 20)
Index 3: ('cherry', 30)
Index 4: Empty
Value for apple: 10
Value for banana 20
Value for cherry: 30
Value for orange None

Hash table after deletion of 'banana':
Hash Table Contents:
Index 0: Empty
Index 1: ('apple', 10)
Index 2: Empty
Index 3: ('cherry', 30)
Index 4: Empty
                </pre>
              </div>
            )}
            
            {currentVideoUrl && (
              <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#ffffff', color: '#374151', border: '1px solid #d1d5db' }}>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>Watch the Algorithm in Action!</h3>
                  <div style={{ position: 'relative', width: '100%', paddingBottom: '56.25%', height: 0 }}>
                      <iframe
                          src={currentVideoUrl}
                          title="Algorithm Visualization"
                          style={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              width: '100%',
                              height: '100%',
                              border: 0,
                          }}
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                          allowFullScreen
                      ></iframe>
                  </div>
              </div>
            )}
          </div>
        </main>
      </div>
      <footer style={{ backgroundColor: 'black', color: 'white', padding: '1rem', textAlign: 'center' }}>
        <p>&#169; 2024 Computer Science and Engineering Department. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default ProgramViewer;
 