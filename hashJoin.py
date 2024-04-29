import random
import string
import sys

from tabulate import tabulate

DISK_BLOCK_SIZE = 8
BUCKET_CAPACITY = 200

# Open the file where you want to redirect the output
output_file = open('hash_join.txt', 'w')

# Save the current state of sys.stdout
original_stdout = sys.stdout

# Set sys.stdout to the file object
sys.stdout = output_file


class Relation:
    def __init__(self, relationName, baseAddress, relationSize=0, reference=None, referenceKeys=None):
        if referenceKeys is None:
            referenceKeys = set()
        if reference is None:
            reference = []
        self.relationSize = relationSize
        self.reference = reference
        self.referenceKeys = referenceKeys
        self.relationName = relationName
        self.baseAddress = baseAddress
        self.statistics = [0]


class DiskBlock:
    def __init__(self, blockSize, data=None):
        if data is None:
            data = []
        self.blockSize = blockSize
        self.data = [None] * self.blockSize
        for idx in range(min(self.blockSize, len(data))):
            self.data[idx] = data[idx]


class Disk:
    def __init__(self):
        self.storage = [None] * 40000  # Increased storage capacity to a specific number
        self.current_position = 0  # Renamed from 'current_position' for clarity
        self.operation_count = 0  # Renamed from 'operation_count' for clear indication of purpose
        self.bucket_start_index = 0  # Renamed from 'bucket_start_index' to clarify usage

    def read(self, index):
        self.operation_count += 1  # Increment the count of I/O operations
        return self.storage[index]  # Return the block at the given index

    def write(self, block, index):
        self.operation_count += 1  # Increment I/O operation count
        self.storage[index] = block  # Store the block at the specified index


class Memory:
    SIZE = 15

    def __init__(self):
        self.baseAddress = 0
        self.storage = [None] * 120
        self.memCache = [None] * 3

    def write(self, blockId, start=0):
        block = DiskBlock(DISK_BLOCK_SIZE, self.storage[start:start + DISK_BLOCK_SIZE])
        disk.write(block, blockId)

    def read(self, disk, blockId, start=0):
        block = disk.read(blockId)
        for i in range(DISK_BLOCK_SIZE):
            self.storage[start + i] = block.data[i]

    def flushToDisk(self):
        for pos in range(self.baseAddress, self.SIZE * 8):
            self.storage[pos] = None


def generateRelation(refBKeys, relation, low, high, relationSize):
    # Initialize the base address for the relation at the current disk cursor position.
    baseAddress = disk.current_position

    # Initialize a list and set for storing the reference tuples and unique keys.
    reference, referenceKeys = [], set()

    # Loop through each block based on the total relation size and disk block size.
    for _ in range(relationSize // DISK_BLOCK_SIZE):
        # Create a new disk block with a predefined size.
        block = DiskBlock(DISK_BLOCK_SIZE)

        # Check if the relation type is 'BC'.
        if relation == "BC":
            blockData = []  # Initialize list to store block data.
            # Fill the block with unique tuples until the block is full.
            while len(blockData) < DISK_BLOCK_SIZE:
                # Generate a random B value within the given range.
                relationB = random.randint(low, high)
                # Ensure the B value is unique within the relation to avoid duplicates.
                if relationB not in referenceKeys:
                    # Generate a random string C value.
                    relationC = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
                    # Add the new B value to the set of reference keys.
                    referenceKeys.add(relationB)
                    # Create a new tuple of (B, C) and add to reference and block data lists.
                    new_tuple = (relationB, relationC)
                    reference.append(new_tuple)
                    blockData.append(new_tuple)
            # Once the block data list is filled, assign it to the block's data attribute.
            block.data = blockData
        else:
            # For other relations like 'AB', generate tuples differently.
            for idx in range(DISK_BLOCK_SIZE):
                # Generate a random string for the second value in the tuple.
                word = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
                # If relation is 'AB', choose B from reference keys, otherwise generate a new random B.
                relationB = random.choice(list(refBKeys)) if relation == "AB" else random.randint(low, high)
                # Add the B value to reference keys and the new tuple to the reference list.
                referenceKeys.add(relationB)
                reference.append((relationB, word))
                # Place the new tuple in the current block at index idx.
                block.data[idx] = (relationB, word)
        # Store the filled block at the current position in disk storage.
        disk.storage[disk.current_position] = block
        # Increment the disk's current position to the next free block.
        disk.current_position += 1

    # Increment the disk's bucket start index by the number of blocks used by this relation.
    disk.bucket_start_index += relationSize // DISK_BLOCK_SIZE

    # Return a new Relation object containing the metadata and the reference data.
    return Relation(relation, baseAddress, relationSize, reference, referenceKeys)



def getHash(key, relationSize):
    hx = 0
    for c in str(key):
        hx = ((hx + ord(c)) * 65599) % relationSize  # Simple and common multiplier in hash functions
    return hx


def getRelationIdx(relation):
    if relation.relationName == "AB":
        return 2
    if relation.relationName == "BC":
        return 0
    return 1


def initializeRelation(relation, numBuckets):
    relation_index = getRelationIdx(relation)
    # Setting disk current_position for this relation's buckets
    disk.current_position = disk.bucket_start_index + relation_index * numBuckets * BUCKET_CAPACITY
    memory.memCache[relation_index] = [0] * numBuckets  # flushToDisk memCache for buckets
    total_blocks = relation.relationSize // DISK_BLOCK_SIZE  # Total blocks to process

    for block_idx in range(total_blocks):
        # Read each block from disk to memory at base address
        memory.read(disk, relation.baseAddress + block_idx, memory.baseAddress)

        for item_index in range(DISK_BLOCK_SIZE):
            item = memory.storage[memory.baseAddress + item_index]
            if item is None:
                continue
            item_key, item_value = item
            bucket_idx = getHash(item_key, numBuckets)
            bucket_address = memory.baseAddress + DISK_BLOCK_SIZE + bucket_idx * DISK_BLOCK_SIZE
            bucket_cache_idx = memory.memCache[relation_index][bucket_idx]

            # Place item in the correct bucket in memory
            memory.storage[bucket_address + bucket_cache_idx % DISK_BLOCK_SIZE] = (item_key, item_value)
            memory.memCache[relation_index][bucket_idx] += 1

            # Check if the bucket is full and needs to be written to disk
            if (bucket_cache_idx + 1) % DISK_BLOCK_SIZE == 0:
                disk_address = disk.current_position + bucket_idx * BUCKET_CAPACITY + bucket_cache_idx // DISK_BLOCK_SIZE
                memory.write(disk_address, bucket_address)
                # Clear the memory slot after writing to disk
                for slot in range(DISK_BLOCK_SIZE):
                    memory.storage[bucket_address + slot] = None

    # Write remaining items in each bucket to disk
    for idx in range(numBuckets):
        remaining_items = memory.memCache[relation_index][idx] % DISK_BLOCK_SIZE
        if remaining_items > 0:
            disk_address = disk.current_position + idx * BUCKET_CAPACITY + memory.memCache[relation_index][
                idx] // DISK_BLOCK_SIZE
            memory.write(disk_address, memory.baseAddress + DISK_BLOCK_SIZE + idx * DISK_BLOCK_SIZE)

    memory.flushToDisk()  # Clear the memory after processing


def hashJoin(relation1, relation2):
    nBuckets = memory.SIZE - 1  # Reserve one block for read operations
    relations = [getRelationIdx(relation1), getRelationIdx(relation2)]
    initialIO = disk.operation_count
    bucketRange = [0] * 2

    # Load buckets into memory if not already cached
    for i, relation in enumerate([relation1, relation2]):
        if memory.memCache[relations[i]] is None:
            initializeRelation(relation, nBuckets)
            relation.statistics[0] = disk.operation_count - initialIO
        bucketRange[i] = relations[i] * nBuckets * BUCKET_CAPACITY

    joinData = []
    disk.current_position = disk.bucket_start_index

    # Join buckets from both relations
    for bucket in range(nBuckets):
        # Read buckets for relation2 into memory
        r2_blocks = (memory.memCache[relations[1]][bucket] + DISK_BLOCK_SIZE - 1) // DISK_BLOCK_SIZE
        for blockIndex in range(r2_blocks):
            memory.read(disk, disk.current_position + bucketRange[1] + bucket * BUCKET_CAPACITY + blockIndex,
                            memory.baseAddress + (blockIndex + 1) * DISK_BLOCK_SIZE)

        # Read buckets for relation1 and join with relation2
        r1_blocks = (memory.memCache[relations[0]][bucket] + DISK_BLOCK_SIZE - 1) // DISK_BLOCK_SIZE
        for block_index in range(r1_blocks):
            memory.read(disk, disk.current_position + bucketRange[0] + bucket * BUCKET_CAPACITY + block_index,
                            memory.baseAddress)
            for tuple in memory.storage[memory.baseAddress + DISK_BLOCK_SIZE:memory.baseAddress + DISK_BLOCK_SIZE +
                                                                             memory.memCache[relations[1]][bucket]]:
                # Get relation2 data from the tuple
                relation2K, relation2V = tuple

                # Iterate over the designated memory block to process possible matches
                for memory_item in memory.storage[memory.baseAddress:memory.baseAddress + DISK_BLOCK_SIZE]:
                    if memory_item is None:
                        break  # Stop processing if a None value is encountered
                    relation1K, relation1V = memory_item
                    if relation2K == relation1K:
                        joinData.append((relation2K, relation1V, relation2V))

    return joinData, disk.operation_count - initialIO


def validateJoinIntegrity(relation1, relation2):
    # Log information to check each joined tuple
    valid = True
    for key, value_from_r1, value_from_r2 in joinData:
        # Validate the tuple by checking if they exist in their respective relation references
        if (key, value_from_r1) not in relation1.reference or (key, value_from_r2) not in relation2.reference:
            print(f"Mismatch Found: Key={key}, R1_Value={value_from_r1}, R2_Value={value_from_r2}")
            valid = False

    # Provide feedback based on the verification process
    if valid:
        print(
            f"Join is correctly formed between {relation1.relationName} and {relation2.relationName} "
            f"with total tuples: {len(joinData)}")
    else:
        print("Errors were found in the join results.")


random.seed(33)
disk = Disk()
memory = Memory()

RBC = generateRelation(None, "BC", 10000, 50000, 5000)
RAB = generateRelation(RBC.referenceKeys, "AB", 0, 0, 1000)
RAB2 = generateRelation(None, "AB2", 20000, 30000, 1200)

joinData, operation_count = hashJoin(RBC, RAB)
validateJoinIntegrity(RBC, RAB)

print(f"Disk IO Count: {operation_count}")

results = [data for data in joinData if data[0] in random.sample(list(RAB.referenceKeys), 20)]
print("\nBC ⨝ AB")

columns = ["B", "C", "A"]
output = tabulate(results, headers=columns, tablefmt="rounded_grid")
print(output)

joinData, operation_count = hashJoin(RBC, RAB2)
validateJoinIntegrity(RBC, RAB2)
print(
    f"Disk IO for pre-computed BC buckets: {operation_count}\n"
    f"Aggregate Disk IO including BC bucket generation: {operation_count + RBC.statistics[0]}"
)

print("\nBC ⨝ AB2")

columns = ["B", "C", "A"]
output = tabulate(joinData, headers=columns, tablefmt="rounded_grid")
print(output)
