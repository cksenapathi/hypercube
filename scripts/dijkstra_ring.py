# Implementation of Dijkstra's Self-Stabilizing Token Ring
import asyncio
from random import randint

class Node:
    def __init__(self, node_id, state, parent=None, child=None, leader=False):
        self.id = node_id
        self.state = state
        self.parent = parent
        self.child = child
        self.leader = leader

    def get_state(self):
        return self.state

    def get_parent_state(self):
        return self.parent.get_state()

    def get_child_state(self):
        return self.child.get_state()

    def __eq__(self, other):
        return (self.id == other.id and
                self.parent == other.parent and
                self.child == other.child)

    async def update_state(self, n):
        if self.leader:
            if self.get_parent_state() == self.state:
                self.state = (self.state + 1) % n # UNKNOWN n!!!!
        else:
            self.state = self.get_parent_state()

    def __repr__(self):
        return "{}: {}".format(self.id, self.state)


class Ring:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.ring = self._create_ring()
        self.leader_node = None

    def _create_ring(self):
        nodes = [Node(node_id=i, state=randint(0, self.num_nodes)) for i in range(self.num_nodes)]
        nodes[randint(0, len(nodes))].leader = True
        # Don't know which one the leader is; Find out!

        for i, n in enumerate(nodes):
            n.parent = nodes[(i+1) % self.num_nodes]
            n.child = nodes[(i-1) % self.num_nodes]

        return tuple(nodes)

    # Stupid stopping condition
    # Should be better way to check stability/convergence
    def run_ring(self, valid_start_state=False):
        i = 0
        while i < 10:
            for n in self.ring:
                asyncio.run(n.update_state(self.num_nodes))
                print(str(n), end=" ")
            print()
            i += 1

ring = Ring(5)
ring.run_ring()


        
