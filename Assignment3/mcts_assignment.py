"""
Please complete and test the implementation of `mcts` and `ConnectFour`.

See the #TODO sections below.

Credits: Much of the code is inspired by https://github.com/aimacode/aima-python/
"""

import copy
import numpy as np
from operator import itemgetter
import pydot
#import random

# GAMES

class GameNode:

    def __init__(self, board, parent = None) -> None:
        self.board = board

    def next_player(self) -> str:
        raise NotImplementedError

    def last_played(self) -> str:
        raise NotImplementedError

    def last_move(self) -> str:
        raise NotImplementedError

    def next_game_node(self, move):
        raise NotImplementedError

    def path(self):
        current = self
        path_back = [current]
        while current.parent is not None:
            path_back.append(current.parent)
            current = current.parent
        return reversed(path_back)

    def is_terminal(self):
        raise NotImplementedError

    def winner(self) -> str:
        raise NotImplementedError

    def available_moves(self):
        raise NotImplementedError

    def utility(self):
        raise NotImplementedError

class MNKNode(GameNode):

    def __init__(self, board, k = 3, parent = None, last_move = None) -> None:
        """
            state (list of lists): Each sub-list is a list of Xs, Os, and -s.
            parent (MNKNode): The parent node that was used to generate this node
        """
        super().__init__(board, parent)

        self.m = len(self.board)
        self.n = len(self.board[0])
        self.k = k

        self.lm = last_move

        self.x_c, self.o_c = self._count()


    def __repr__(self) -> str:
        res = ""
        for row in self.board:
            res += " ".join(row) + "\n"
        return res

    def _count(self):
        board_arr = np.asarray(self.board)
        x_c = np.sum(board_arr == 'X')
        o_c = np.sum(board_arr == 'O')
        return x_c, o_c

    def next_player(self) -> str:

        # X goes first
        if self.x_c == self.o_c:
            return 'X'
        else:
            return 'O'


    def last_played(self) -> str:
        p = self.next_player();
        if p == 'X':
            return 'O'
        else:
            return 'X'

    def last_move(self):
        return self.lm

    def next_game_node(self, move):

        x, y, p = move
        assert p == self.next_player()
        assert self.board[x][y] == '-'
        assert not self.is_terminal()
        
       
        new_board = copy.deepcopy(self.board)
        new_board[x][y] = p

        return MNKNode(board=new_board, k=self.k, parent = self, last_move = copy.deepcopy(move))     

    def available_moves(self):

        p = self.next_player();
        moves = []

        for i in range(self.m):
            for j in range(self.n):
                if self.board[i][j] == '-':
                    moves.append((i, j, p))

        return moves

    def _is_winner(self, p):

        return self._atleastk(p)

    def winner(self) -> str:

        if self.is_terminal():
            if self._is_winner('X'):
                print('terminal - winner: X')
                return 'X'
            elif self._is_winner('O'):
                print('terminal - winner: O')
                return 'O'

            else:
                print('terminal - no winner')
                return None

    def _atleastk_line(self, p, begin_x, begin_y, delta_x, delta_y):
        max_count = 0
        x = begin_x
        y = begin_y

        while (x >= 0) and (x < self.m) and (y >= 0) and (y < self.n):

            if self.board[x][y] == p:
                max_count += 1
            else:
                max_count = 0

            if max_count >= self.k:
                return True

            x += delta_x
            y += delta_y

        return False

    def _atleastk(self, p):

        one_found = False


        # cols        
        for begin_y in range(0, self.n):
            one_found = self._atleastk_line(p, begin_x = self.m-1, begin_y = begin_y, delta_x = -1, delta_y = 0)
            if one_found:
                return True
        # rows
        for begin_x in range(self.m-1, -1, -1):
            one_found = self._atleastk_line(p, begin_x, begin_y = 0, delta_x = 0, delta_y = 1)
            if one_found:
                return True

        # NE diag (dx = -1, dy = +1)
        ## West edge (begin_y = 0)
        for begin_x in range(self.m-1, -1, -1):
            one_found = self._atleastk_line(p, begin_x, begin_y = 0, delta_x = -1, delta_y = 1)
            if one_found:
                return True

        ## South edge (begin_x = m-1)
        for begin_y in range(self.n):
            one_found = self._atleastk_line(p, begin_x = self.m-1, begin_y = begin_y, delta_x = -1, delta_y = 1)
            if one_found:
                return True

        # NW diag (dx = -1, dy = -1)
        ## East edge (begin_y = n-1)
        for begin_x in range(self.m-1, -1, -1):
            one_found = self._atleastk_line(p, begin_x, begin_y = self.n-1, delta_x = -1, delta_y = -1)
            if one_found:
                return True

        ## South edge (begin_x = m-1)
        for begin_y in range(self.n):
            one_found = self._atleastk_line(p, begin_x = self.m-1, begin_y = begin_y, delta_x = -1, delta_y = -1)
            if one_found:
                return True

        return False

    def _is_full(self):
        return self.x_c + self.o_c == self.m * self.n

    def is_terminal(self):


        if self._is_full():
            return True

        # Is X a winner?
        if self._is_winner('X'):
            return True

        # Is O a winner?
        if self._is_winner('O'):
            return True
        
        return False

    def utility(self):
        if self._is_winner('X'):
            return 1
        elif self._is_winner('O'):
            return -1
        elif self._is_full():
            return 0
        else:
            return None

class DictGameNode(GameNode):

    _moves = None # A dictionary of moves (list) available at each state; a move is also the name of the next state
    _terminal_nodes = None # Key value pairs for the terminal states

    def __init__(self, board, np, parent = None, last_move = None) -> None:
        """
            state (list of lists): Each sub-list is a list of Xs, Os, and -s.
            parent (MNKNode): The parent node that was used to generate this node
        """
        super().__init__(board, parent)

        self.np = np
        self.lm = last_move

    def __repr__(self) -> str:
        return str(self.board)

    def next_player(self) -> str:
        return self.np

    def last_played(self) -> str:
        nxt = self.next_player();
        if nxt == 'X':
            return 'O'
        else:
            return 'X'

    def last_move(self):
        return self.lm

    def next_game_node(self, move):
        # move is the name of the next board
        _np = None
        if self.np == 'X':
            _np = 'O'
        else:
            _np = 'X'

        return DictGameNode(board=move, parent = self, np = _np, last_move = move)

    def available_moves(self):
        if self.is_terminal():
            return []
        else:
            return DictGameNode._moves[self.board]

    def winner(self) -> str:
        if self.is_terminal():
            util = DictGameNode._terminal_nodes[self.board]
            if util == 1:
                return 'X'
            elif util == -1:
                return 'O'
            else:
                return None
        else:
            return None


    def is_terminal(self):
        return self.board in DictGameNode._terminal_nodes

    def utility(self):

        if self.is_terminal():
            return DictGameNode._terminal_nodes[self.board]
        else:
            return None

class ConnectFour(MNKNode):

    def __init__(self, board, parent=None, last_move=None) -> None:
        super().__init__(board, 4, parent, last_move)

    def next_game_node(self, move):
        x, y, p = move
        assert p == self.next_player()
        assert self.board[x][y] == '-'
        assert not self.is_terminal()

        new_board = copy.deepcopy(self.board)
        new_board[x][y] = p
        return ConnectFour(board=new_board, parent = self, last_move = copy.deepcopy(move))


    def available_moves(self):
        # TODO: Complete the implementation.
        # Return a list of moves. A single move is (x, y, p) where x and y are the coordinates and p is the player.
        # See MNKNode as an example.
        p = self.next_player();
        moves = []

        for i in range(self.m):
            for j in range(self.n):
                if self.board[i][j] == '-':
                    moves.append((i, j, p))

        return moves
        
        
        
# SEARCH ALGORITHMS

def minmax_decision(game_node):

    def max_value(gn):

        if gn.is_terminal():
            return gn.utility()
        v = -np.inf
        for move in gn.available_moves():
            v = max(v, min_value(gn.next_game_node(move)))
        return v

    def min_value(gn):

        if gn.is_terminal():
            return gn.utility()
        v = np.inf
        for move in gn.available_moves():
            v = min(v, max_value(gn.next_game_node(move)))
        return v

    p = game_node.next_player();
    moves = game_node.available_moves()

    if p == "X":
        return [(move, min_value(game_node.next_game_node(move))) for move in moves]
    else:
        return [(move, max_value(game_node.next_game_node(move))) for move in moves]


def alpha_beta_search(game_node):

    def max_value(gn, alpha, beta):

        if gn.is_terminal():
            return gn.utility()
        v = -np.inf
        for move in gn.available_moves():
            v = max(v, min_value(gn.next_game_node(move), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(gn, alpha, beta):

        if gn.is_terminal():
            return gn.utility()
        v = np.inf
        for move in gn.available_moves():
            v = min(v, max_value(gn.next_game_node(move), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:

    p = game_node.next_player();
    moves = game_node.available_moves()

    alpha = -np.inf
    beta = np.inf

    if p == "X":
        return [(move, min_value(game_node.next_game_node(move), alpha, beta)) for move in moves]
    else:
        return [(move, max_value(game_node.next_game_node(move), alpha, beta)) for move in moves]


class MCNode:

    def __init__(self, gn, N = 0, U = 0, parent = None, children = []) -> None:
        self.gn = gn
        self.N = N
        self.U = U
        self.parent = parent
        self.children = children

    def __repr__(self):
        return str(self.gn) + "\n" + "U/N: {}/{}".format(self.U, self.N)

def ucb1(mcnode, C = 1.4):
    if mcnode.N == 0:
        return np.inf
    else:
        return (mcnode.U/mcnode.N) + C * np.sqrt(np.log(mcnode.parent.N)/mcnode.N)

def kids(mcnode):
    game_state = mcnode.gn
    moves = game_state.available_moves()
    x_count = 0
    o_count = 0
    B = np.matrix(game_state.board)
    rows,cols = B.shape
    for y in range(rows):
        for x in range(cols):
            if B[y,x] == 'X':
                x_count += 1
            if B[y,x] == 'O':
                o_count += 1

    to_add = 'X' if x_count != o_count else 'O'
    children_ = []
    for m in moves:
        x,y,p = m
        #assert p == self.next_player()
        #assert self.board[x][y] == '-'
        #assert not self.is_terminal()
        
        new_board = copy.deepcopy(game_state.board)
        new_board[x][y] = p
        c = ConnectFour(board=new_board, last_move = copy.deepcopy(m))
        
        
        
        #c = game_state.next_game_node(m)
        child = MCNode(c)
        child.parent = mcnode
        if not child in mcnode.children:
            children_.append(child)
    return children_
    
    

def mcts(root_gn, rs, max_iter=10):
    this_state = root_gn
    def select_leaf(mcnode):
        if mcnode.children == []:
            return mcnode
        else:
            ucb_vals = [ucb1(c,C=1.4) for c in mcnode.children]
            child_val_dict = dict(zip(ucb_vals,mcnode.children))
            max_val = max(ucb_vals)
            candidates = [c for c in mcnode.children if child_val_dict[max_val] == c]
            next_leaf = rs.choice(candidates)
            return select_leaf(next_leaf)
        # TODO: Complete the implementation.
        # Select a leaf node in the search tree.
        # Use UCB1 value as the criteria.
        # Hint: My implementation is recursive. Here is the idea:
        # If mcnode has children, choose its max ucb1 child, and recurse.
        # If mcnode has no children, return it.
        raise NotImplementedError
        


    def expand(mcnode_):
        children_ = kids(mcnode_)
        mcnode_.children = children_
        game_state = mcnode_.gn
        node_N     = mcnode_.N
        if game_state.is_terminal():
            return mcnode_
        elif node_N == None:
            return mcnode_
        else:
            if children_ == []:
                return mcnode
            else:
                for c in children_:
                    backprop(c,c.gn)
                ri = rs.choice(len(children_))
                return children_[ri]
        raise NotImplementedError

    def simulate(mcnode):
        current_gn = mcnode.gn
        
        while not current_gn.is_terminal():
            children_ = kids(mcnode)
            mcnode.children = children_
            ri = rs.choice(len(children_))
            child = children_[ri]
            current_gn = child.gn
            mcnode = MCNode(current_gn)
        return current_gn.utility()

    def backprop(current_mcnode, util):
        current_mcnode

        while current_mcnode is not None:
            current_mcnode.N += 1
            gn = current_mcnode.gn
            player = gn.last_played()
            if (util == -1 and player == 'O') or (util == +1 and player == 'X'):
                current_mcnode.U += 1
            elif util == 0: # Draw; we need to change it in the future assignments.
                current_mcnode.U += 0.5
            current_mcnode = current_mcnode.parent
            if gn.is_terminal():
                break

    root_mcnode = MCNode(root_gn)
    for _ in range(max_iter):
        leaf_mcnode = select_leaf(root_mcnode)
        child_mcnode = expand(leaf_mcnode)
        util = simulate(child_mcnode)
        backprop(child_mcnode, util)
    return root_mcnode

### Edited mcts_move to not iterate over root_mcnode.children if == None
# Original:
def mcts_move(root_mcnode):
    best_mcnode = max([(child, child.N) for child in root_mcnode.children], key=itemgetter(1))[0]
    #print("{}/{} = {:.2f}".format(best_mcnode.U, best_mcnode.N, best_mcnode.U/best_mcnode.N))
    return best_mcnode.gn.last_move()

# GAME PLAY

def random_play(initial_node, seed = None):
    random_state = np.random.RandomState(seed)

    current_node = initial_node

    while not current_node.is_terminal():
        nxt = current_node.next_player();
        print("It's {}'s turn.".format(nxt))
        moves = current_node.available_moves();
        ri = random_state.randint(low=0, high=len(moves))
        chosen_move = moves[ri]
        print("Chosen move {}".format(str(chosen_move)))
        print()
        current_node = current_node.next_game_node(chosen_move)


    winner = current_node.winner()

    if winner is not None:
        print("Winner is {}.".format(winner))
    else:
        print("Draw.")


## mine:
def mcts_player(gn,rs=0, max_iter=10):

    if gn.is_terminal():       
        return mcts_move(root_mcnode)
    else:
        root_mcnode = mcts(gn, np.random.RandomState(0), max_iter=max_iter)
        return mcts_move(root_mcnode)

def maxplayer(gn, algo=minmax_decision):
    res = algo(gn)
    chosen_move_util = max(res, key = itemgetter(1))

    util = chosen_move_util[1]
    if util == 1:
        print("X has a winning strategy")
    elif util == -1:
        print("O has a winning strategy")

    return chosen_move_util[0]

def minplayer(gn, algo=minmax_decision):
    res = algo(gn)
    chosen_move_util = min(res, key = itemgetter(1))

    util = chosen_move_util[1]
    if util == 1:
        print("X has a winning strategy.")
    elif util == -1:
        print("O has a winning strategy.")

    return chosen_move_util[0]

def randplayer(gb, rs):
    moves = gb.available_moves()
    ri = rs.randint(low=0, high=len(moves))
    return moves[ri]

def firstmoveplayer(gb):
    moves = gb.available_moves()
    return moves[0]

def human_player(gb, p):
    move = input("Your move, x, y separated by comma:")
    move = move.split(',')
    move[0] = int(move[0])
    move[1] = int(move[1])
    move.append(p)
    return tuple(move)

def game_play(initial_node, x_player, o_player):
    current_gn = initial_node

    while not current_gn.is_terminal():
        print(current_gn)
        p = current_gn.next_player();
        print("It's {}'s turn.".format(p))

        if p == 'X':
            chosen_move = x_player(current_gn)
        else:
            chosen_move = o_player(current_gn)

        print("Chosen move {}.".format(str(chosen_move)))
        print()
        current_gn = current_gn.next_game_node(chosen_move)

    print("\nGame ended.")
    print(current_gn)


    winner = current_gn.winner()

    if winner is not None:
        print("Winner is {}.".format(winner))
    else:
        print("Draw.")

# VISUALIZATION

def dot_graph(mc_root):
    frontier = [mc_root]

    graph = pydot.Dot("my_graph", graph_type="graph")

    while len(frontier) > 0:
        node = frontier.pop(0)

        graph.add_node(pydot.Node(str(id(node)), label=str(node)))

        if node.parent is not None:
            graph.add_edge(pydot.Edge(str(id(node.parent)), str(id(node))))

        if node.children is []:
            frontier.extend(node.children)

    return graph