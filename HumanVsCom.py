'''
-----------------------Authors-------------------------
              Nayanika Ghosh - 4191 4976
                Disha Nayar - 6199 1035
-------------------------------------------------------
'''
import numpy as np
import pickle

class PlayHumanvsCom():
    def __init__(self):
        self.play()

    def predict_LR(self, board):
        board_lr = np.append(1, board)
        weights = np.loadtxt('/Users/nghosh/Desktop/Deeplearning/LRWeights.txt')
        a = weights[:1, :].T
        y0_pred = np.dot(board_lr, weights[:1, :].T)
        y1_pred = np.dot(board_lr, weights[1:2, :].T)
        y2_pred = np.dot(board_lr, weights[2:3, :].T)
        y3_pred = np.dot(board_lr, weights[3:4, :].T)
        y4_pred = np.dot(board_lr, weights[4:5, :].T)
        y5_pred = np.dot(board_lr, weights[5:6, :].T)
        y6_pred = np.dot(board_lr, weights[6:7, :].T)
        y7_pred = np.dot(board_lr, weights[7:8, :].T)
        y8_pred = np.dot(board_lr, weights[8:9, :].T)
        y_pred = np.array([y0_pred, y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred])
        max_index = y_pred.tolist().index(np.max(y_pred))
        y_pred_copy = y_pred
        while (board[max_index] != 0):
            y_pred_copy[max_index] = 0
            max_index = y_pred_copy.tolist().index(np.max(y_pred_copy))
        b = np.zeros_like(y_pred)
        b[max_index] = 1
        return b

    def select_regressor(self):
        print("Enter 1 to play against Linear Regressor")
        print("Enter 2 to play against KNN Regressor")
        print("Enter 3 to play against MLP Regressor")
        user_input = input()
        return user_input

    def take_input(self):
        '''
        This function takes a cell number as input from the user
        the cell number needs to be between 1 to 9 and should be empty
        :return: string
        '''
        print("Enter cell number")
        user_input = input()
        return user_input

    def decode(self, cell_value):
        '''
        This function decodes -1 as O and 1 as X
        It is used as a helper function to display the board
        :param cell_value: takes the value 0,-1 or 1
        :return: None
        '''
        if (cell_value == 0):
            return '  '
        elif (cell_value == -1):
            return 'O '
        else:
            return 'X '

    def display_board(self, board=[]):
        '''
        This function displays the board to the user
        :param board: the current state of the board
        :return: None
        '''
        print(self.decode(board[0]) + '|' + self.decode(board[1]) + '|' + self.decode(board[2]))
        print("-- -- --")
        print(self.decode(board[3]) + '|' + self.decode(board[4]) + '|' + self.decode(board[5]))
        print("-- -- --")
        print(self.decode(board[6]) + '|' + self.decode(board[7]) + '|' + self.decode(board[8]))
        print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

    def play(self):
        '''
        This function implements the logic to play tic tac toe with human
        :return: None
        '''
        print("Lets Play!!!!")
        regressor = self.select_regressor()
        print("Enter a cell number (1 to 9)")
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.display_board(board)
        while (board.count(0) != 0):
            input_pos = self.take_input()
            while ((int(input_pos) < 1 or int(input_pos) > 9) or board[int(input_pos) - 1] != 0):
                print("Please enter a valid input")
                input_pos = self.take_input()
            board[int(input_pos) - 1] = 1
            self.display_board(board)
            # check if human won
            if ((board[0], board[1], board[2]) == (1, 1, 1) or (board[3], board[4], board[5]) == (1, 1, 1) or (
                    board[6], board[7], board[8]) == (1, 1, 1) or (board[0], board[3], board[6]) == (1, 1, 1) or (
                    board[1], board[4], board[7]) == (1, 1, 1) or (board[2], board[5], board[8]) == (1, 1, 1) or (
                    board[0], board[4], board[8]) == (1, 1, 1) or (board[2], board[4], board[6]) == (1, 1, 1)):
                print("human won")
                break
            if (board.count(0) == 0):
                print("Its a Draw!!")
                break
            while(True):
                if (int(regressor) == 1):
                    # predicts computers next move
                    # predict using Linear regression
                    y_pred = self.predict_LR(board)
                    pos_op = [i for i, val in enumerate(y_pred) if val == 1]
                    break
                elif(int(regressor) == 2):
                    # predicts computers next move
                    # predict using KNN
                    KNN_model = pickle.load(open('/Users/nghosh/Desktop/Deeplearning/knnweights', 'rb'))
                    y_pred = KNN_model.predict(np.asarray(board).reshape(1, -1))
                    y_pred = np.where(y_pred > 0.5, 1, 0)
                    pos_op = [i for i, val in enumerate(y_pred[0]) if val == 1]
                    break
                elif(int(regressor) == 3):
                    # predicts computers next move
                    # predict using MLP
                    MLP_model = pickle.load(open('/Users/nghosh/Desktop/Deeplearning/MLPWeights', 'rb'))
                    y_pred = MLP_model.predict(np.asarray(board).reshape(1, -1))
                    y_pred = np.where(y_pred > 0.5, 1, 0)
                    pos_op = [i for i, val in enumerate(y_pred[0]) if val == 1]
                    break
                else:
                    print("Enter a valid regressor!")

            predicted_move = np.random.choice(a=np.array(pos_op))
            board[predicted_move] = -1
            self.display_board(board)
            # check if computer won
            if ((board[0], board[1], board[2]) == (-1, -1, -1) or (board[3], board[4], board[5]) == (-1, -1, -1) or (
                    board[6], board[7], board[8]) == (-1, -1, -1) or (board[0], board[3], board[6]) == (-1, -1, -1) or (
                    board[1], board[4], board[7]) == (-1, -1, -1) or (board[2], board[5], board[8]) == (-1, -1, -1) or (
                    board[0], board[4], board[8]) == (-1, -1, -1) or (board[2], board[4], board[6]) == (-1, -1, -1)):
                print("comp won")
                break

if __name__ == "__main__":
    play = PlayHumanvsCom()






