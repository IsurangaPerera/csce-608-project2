import math
import sys
import random
from bisect import bisect_right
from bisect import bisect_left
from bisect import insort

# Open the file where you want to redirect the output
output_file = open('bplustree_output.txt', 'w')

# Save the current state of sys.stdout
original_stdout = sys.stdout

# Set sys.stdout to the file object
sys.stdout = output_file


def printBanner(message):
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════════════════╗",
        f"║ {message.center(88)} ║",
        "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
    ]
    for line in lines:
        print(line)


class Node:
    def __init__(self, limit, isLeaf=False):
        self.limit = limit
        self.records = []
        self.successors = []
        self.isLeaf = isLeaf
        self.isRoot = False
        self.followingNode = None
        self.previous = None
        self.predecessor = None

    def add(self, key):
        if not self.isLeaf:
            return self.addToInternal(key)
        else:
            return self.addToLeaf(key)

    def addToInternal(self, key):
        # Determine the correct child to add the key into by using a binary search to find the insertion point.
        insert_position = bisect_right(self.records, key)
        assumedKey, new_child_node = self.successors[insert_position].add(key)

        # Check if a new node was created during the add, which would need to be accommodated in this internal node.
        if new_child_node:
            print("Internal nodes before adding new element : ", self.records)
            # Find the location to add the promoted key that came up from the child node.
            key_position = bisect_right(self.records, assumedKey)

            # Insert the promoted key and the new node at their respective positions.
            self.records.add(key_position, assumedKey)
            self.successors.add(key_position + 1, new_child_node)
            new_child_node.predecessor = self

            print("Internal nodes after adding new element : ", self.records)

            # If the current node exceeds the limit, it needs to be split.
            if len(self.records) > self.limit:
                return self.splitInternal()

        # Return None, None if no new node needs to be created or promoted.
        return None, None

    def splitLefNode(self):
        # Calculate the middle index to split the leaf node records evenly
        # Create a new leaf node that will store the second half of records
        right = Node(self.limit, True)
        right.previous = self
        right.followingNode = self.followingNode
        right.records = self.records[len(self.records) // 2:]

        # Adjust the linked list pointers to incorporate the new leaf node
        if self.followingNode:
            self.followingNode.previous = right
        self.followingNode = right

        # Retain only the first half of records in the current leaf node
        self.records = self.records[:len(self.records) // 2]

        # Output the state of the leaf nodes after the split for debugging
        print("Splitting the leaf node :")
        print(f"Keys in the current node : {self.records}")
        print(f"Keys in the new right sibling node : {right.records}")

        # Return the smallest key from the new node along with the new node itself
        return right.records[0], right

    def addToLeaf(self, key):
        # Print the initial state of the leaf node for debugging
        print(f"Leaf nodes before the operation : {self.records}")

        # Check if the key already exists in the node to avoid duplicates
        if key in self.records:
            return None, None

        # Insert the key while maintaining sorted order
        insort(self.records, key)

        # Print the state of the leaf node after insertion for debugging
        print(f"Leaf nodes after the operation : {self.records}")
        print(self.records)

        # Check if the node exceeds the limit, requiring a split
        if len(self.records) > self.limit:
            return self.splitLefNode()

        # Return None if no split is needed
        return None, None

    def splitInternal(self):
        # Calculate the index for splitting the node
        splitPoint = len(self.records) // 2
        assumedKey = self.records[splitPoint]

        # Create a new internal node for the second half of the records and successors
        new_sibling = Node(self.limit, False)
        new_sibling.records = self.records[splitPoint + 1:]
        new_sibling.successors = self.successors[splitPoint + 1:]

        # Update the current node to only contain the first half of the records and successors
        self.records = self.records[:splitPoint]
        self.successors = self.successors[:splitPoint + 1]

        # Assign the new predecessor for the successors moved to the new sibling node
        for child in new_sibling.successors:
            child.predecessor = new_sibling

        # Debugging outputs to trace the state of the internal node after the split
        print("Internal node split completed : ")
        print(f"Keys in the current node: {self.records}")
        print(f"Keys in the new sibling node: {new_sibling.records}")

        # Return the key to be promoted to the predecessor and the new sibling node
        return assumedKey, new_sibling

    def delete(self, key):
        # Decide on the operation based on the node type: leaf or internal
        if self.isLeaf:
            print(f"Leaf nodes before : {self.records}")

            if key not in self.records:
                return None, None

            self.records.remove(key)
            print(f"Leaf nodes after : {self.records}")

            if self.isRoot or len(self.records) >= math.ceil((self.limit + 1) / 2):
                return None, None

            # Attempt to borrow a key from a sibling or merge if necessary
            return self.attemptBorrow()
        else:
            index = bisect_right(self.records, key)
            direction, updated_key = self.successors[index].delete(key)
            if not updated_key:
                return None, None

            print(f"Internal nodes before the operation : {self.records}")
            if direction >= 0:
                # Update key at the specified index if there was no structural change
                self.records[index - direction] = updated_key
            else:
                # Handle different merging or shrinking scenarios
                # Remove key and child based on direction
                if direction == -1:
                    # Straight removal from the current index
                    self.records.pop(index)
                    self.successors = self.successors.pop(index)
                else:
                    # Removal from an adjusted index (when merging with previous sibling)
                    self.records.pop(index - 1)
                    self.successors = self.successors.pop(index - 1)
            print(f"Internal nodes after the operation : {self.records}")

            # Check if current node still meets the minimum fill requirement
            return self.verify_node_fill(direction)

    def attemptBorrow(self):
        # Check for key borrowing or merging scenarios with siblings
        if self.followingNode and self.predecessor == self.followingNode.predecessor and len(self.followingNode.records) > math.ceil((self.limit + 1) / 2):
            return self.borrowNext()
        elif self.previous and self.predecessor == self.previous.predecessor and len(self.previous.records) > math.ceil(
                (self.limit + 1) / 2):
            return self.borrowPrevious()
        elif self.followingNode and self.predecessor == self.followingNode.predecessor:
            return self.mergeNext()
        elif self.previous and self.predecessor == self.previous.predecessor:
            return self.mergePrevious()
        return None, None

    def borrowNext(self):
        print(f"Leaf nodes before borrowing from the sibling : {self.followingNode.records}")
        self.records.append(self.followingNode.records.pop(0))
        print(f"Leaf nodes after borrowing from the sibling : {self.records}")
        return 0, self.followingNode.records[0] if self.followingNode.records else None

    def borrowPrevious(self):
        print(f"Leaf nodes before borrowing from the sibling : {self.previous.records}")
        self.records.add(0, self.previous.records.pop())
        print(f"Leaf nodes after borrowing from the sibling : {self.records}")
        return 1, self.records[0]

    def mergeNext(self):
        print(f"Leaf nodes before merging with sibling : {self.followingNode.records}")
        self.records.extend(self.followingNode.records)
        self.followingNode = self.followingNode.followingNode
        if self.followingNode:
            self.followingNode.previous = self
        print(f"Leaf nodes after merging : {self.records}")
        return -1, self.records[0]

    def mergePrevious(self):
        print(f"Leaf nodes before merging with sibling : {self.previous.records}")
        self.previous.records.extend(self.records)
        self.previous.followingNode = self.followingNode
        if self.followingNode:
            self.followingNode.previous = self.previous
        print(f"Leaf nodes after merging : {self.previous.records}")
        return -2, self.previous.records[0]

    def verify_node_fill(self, direction):
        # Check if the node has enough records to meet the minimum requirement
        if len(self.records) >= math.ceil((self.limit + 1) / 2):
            return None, None
        else:
            # Return adjustment direction and the new minimum key if underflow occurs
            return direction, self.records[0] if self.records else (None, None)

    def search(self, key):
        if self.isLeaf:
            found_keys = [key] if key in self.records else []
            print(f"Key Found : {found_keys}\n" if found_keys else "Key Not Found!\n")
            return found_keys
        else:
            return self.searchSuccessors(key)

    def searchSuccessors(self, key):
        # Use binary search to find the right child to search in, which is more efficient than linear search
        successorIndex = self.findSuccessor(key)
        return self.successors[successorIndex].search(key)

    def findSuccessor(self, key):
        # Using bisect_right to find the insertion point gives us the child index for non-leaf nodes
        return bisect_right(self.records, key) - 1

    def searchRange(self, begin, end):
        if self.isLeaf:
            return self.searchData(begin, end)
        else:
            return self.successors[bisect_left(self.records, begin)].searchRange(begin, end)

    def searchData(self, begin, end):
        result = []
        segment = self
        while segment:
            for record in segment.records:
                if record > end:
                    break
                if begin <= record:
                    result.append(record)
            segment = segment.followingNode

        print(
            f"{len(result)} Keys Found : {result}\n" if result else "No Keys Found\n")
        return result


class Tree:
    def __init__(self, degree, isSparseIndex=False):
        self.degree = degree
        self.isSparseIndex = isSparseIndex
        self.maxKeys = math.ceil(degree / 2) if isSparseIndex else degree
        self.rootNode = Node(self.maxKeys, True)

    def addKey(self, key):
        print(f"Inserting the key {key} into the B+ tree")
        assumedKey, child = self.rootNode.add(key)

        # Check if a new child node was created that needs to be handled at this rootNode level
        if child:
            # Create a new rootNode node because the old rootNode has been split
            node = Node(self.maxKeys, False)
            node.isRoot = True
            node.records = [assumedKey]
            node.successors = [self.rootNode, child]

            # Update the predecessor pointers
            self.rootNode.predecessor = child.predecessor = node

            # Update the tree's rootNode pointer to the new rootNode node
            self.rootNode = node
            print(f"Created a new rootNode with records : {self.rootNode.records}")

        print("Insertion completed successfully \n")

    def buildTree(self, key_list):
        for key in key_list:
            self.addKey(key)

    def removeKey(self, key):
        print(f"Deleting the key : {key} from the B+ tree")
        _, node = self.rootNode.delete(key)

        # If a new rootNode is returned, update the tree's rootNode pointer
        if node:
            self.rootNode = node
            self.rootNode.predecessor = None  # Ensure the new rootNode has no predecessor
            print(f"Updated the rootNode with records : {node.records}")

        print("Deletion completed successfully \n")

    def findKey(self, key):
        print(f"Searching the key : {key}.")
        result = self.rootNode.search(key)
        return result

    def findRange(self, begin, end):
        print(f"Searching range {begin} to {end}.")
        results = self.rootNode.searchRange(begin, end)
        return results


def genKey(low=100000, high=200000):
    return random.randint(low, high)


def randOperations(tree, records, n=5):
    # Insert random new records into the tree
    for _ in range(n):
        tree.addKey(genKey())

    # Remove existing records randomly selected from the provided key list
    for _ in range(n):
        tree.removeKey(random.choice(records))


def build_trees(records):
    printBanner("Populating Order 13 Dense B+ Tree")
    dt13 = Tree(13, isSparseIndex=False)
    dt13.buildTree(records)

    printBanner("Populating Order 24 Dense B+ Tree")
    dt24 = Tree(24, isSparseIndex=False)
    dt24.buildTree(records)

    printBanner("Populating Order 13 Sparse B+ Tree")
    st13 = Tree(13, isSparseIndex=True)
    st13.buildTree(records)

    printBanner("Populating Order 24 Sparse B+ Tree")
    st24 = Tree(24, isSparseIndex=True)
    st24.buildTree(records)

    return dt13, dt24, st13, st24


def executeExperiments():
    random.seed(33)
    records = random.sample(range(100000, 200000), 10000)
    dense13, dense24, sparse13, sparse24 = build_trees(records)

    printBanner("Applying 2 Randomly Generated Insertions on Order 13 Dense B+ Tree")
    dense13.addKey(genKey())
    dense13.addKey(genKey())

    printBanner("Applying 2 Randomly Generated Insertions on Order 24 Dense B+ Tree")
    dense24.addKey(genKey())
    dense24.addKey(genKey())

    printBanner("Applying 2 Randomly Generated Deletions on Order 13 Sparse B+ Tree")
    randKeyRemove(sparse13, records)

    printBanner("Applying 2 Randomly Generated Deletions on Order 24 Sparse B+ Tree")
    randKeyRemove(sparse24, records)

    printBanner("Executing Random Operations on Order 13 Dense B+ Tree")
    randOperations(dense13, records)

    printBanner("Executing Random Operations on Order 24 Dense B+ Tree")
    randOperations(dense24, records)

    printBanner("Executing Random Operations on Order 13 Sparse B+ Tree")
    randOperations(sparse13, records)

    printBanner("Executing Random Operations on Order 24 Sparse B+ Tree")
    randOperations(sparse24, records)

    printBanner("Performing Random Searches on Order 13 Dense B+ Tree")
    randSearch(dense13)
    printBanner("Performing Random Searches on Order 24 Dense B+ Tree")
    randSearch(dense24)

    printBanner("Performing Random Searches on Order 13 Sparse B+ Tree")
    randSearch(sparse13)

    printBanner("Performing Random Searches on Order 24 Sparse B+ Tree")
    randSearch(sparse24)


def randKeyRemove(tree, records):
    tree.removeKey(random.choice(records))
    tree.removeKey(random.choice(records))


def randSearch(tree):
    for _ in range(5):
        tree.findKey(genKey())  # Assuming findKey() searches for a single key
    for _ in range(5):
        begin = genKey()
        tree.findRange(begin, begin + random.randint(1, 100))  # Assuming findRange() searches for a range of records


executeExperiments()
