import numpy as np
import os

class TicTacToe:
    def __init__(self):
        self.board = np.array([' '] * 9)  # Игровое поле 3x3
        self.current_player = 'X'  # X начинает первым
        self.game_over = False
        self.winner = None
        
    def print_board(self):
        """Отображает игровое поле"""
        os.system('cls' if os.name == 'nt' else 'clear')  # Очистка экрана
        print("\n  Крестики-нолики с AI (Minimax)")
        print("  -----------------")
        print(f"  Сейчас ходит: {self.current_player}")
        print()
        print("    0   1   2")
        print("  -------------")
        for i in range(3):
            print(f"{i} | {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} |")
            print("  -------------")
        print()
    
    def available_moves(self):
        """Возвращает список доступных ходов"""
        return [i for i, cell in enumerate(self.board) if cell == ' ']
    
    def make_move(self, position, player=None):
        """Выполняет ход на указанную позицию"""
        if player is None:
            player = self.current_player
            
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False
    
    def check_winner(self):
        """Проверяет, есть ли победитель"""
        # Все выигрышные комбинации
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Горизонтали
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Вертикали
            [0, 4, 8], [2, 4, 6]              # Диагонали
        ]
        
        for combo in win_combinations:
            if (self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' '):
                self.winner = self.board[combo[0]]
                self.game_over = True
                return self.winner
        
        # Проверка на ничью
        if ' ' not in self.board:
            self.game_over = True
            self.winner = 'Ничья'
            return 'Ничья'
        
        return None
    
    def switch_player(self):
        """Переключает текущего игрока"""
        self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    def reset_game(self):
        """Сбрасывает игру к начальному состоянию"""
        self.board = np.array([' '] * 9)
        self.current_player = 'X'
        self.game_over = False
        self.winner = None


class MinimaxAI:
    def __init__(self, player='O'):
        self.player = player  # AI играет за 'O'
        self.opponent = 'X' if player == 'O' else 'O'
    
    def evaluate(self, board):
        """Оценка текущего состояния доски"""
        # Проверяем выигрышные комбинации
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Горизонтали
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Вертикали
            [0, 4, 8], [2, 4, 6]              # Диагонали
        ]
        
        for combo in win_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]]:
                if board[combo[0]] == self.player:
                    return 10  # AI выигрывает
                elif board[combo[0]] == self.opponent:
                    return -10  # Противник выигрывает
        
        return 0  # Ничья или игра продолжается
    
    def minimax(self, board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
        """Алгоритм Minimax с альфа-бета отсечением"""
        score = self.evaluate(board)
        
        # Если игра окончена, возвращаем оценку
        if score == 10 or score == -10:
            return score - depth if score > 0 else score + depth
        
        # Если нет доступных ходов - ничья
        available_moves = [i for i, cell in enumerate(board) if cell == ' ']
        if not available_moves:
            return 0
        
        if is_maximizing:
            # Ход AI (максимизируем оценку)
            max_eval = -float('inf')
            for move in available_moves:
                board[move] = self.player
                eval_score = self.minimax(board, depth + 1, False, alpha, beta)
                board[move] = ' '
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Альфа-бета отсечение
            return max_eval
        else:
            # Ход противника (минимизируем оценку)
            min_eval = float('inf')
            for move in available_moves:
                board[move] = self.opponent
                eval_score = self.minimax(board, depth + 1, True, alpha, beta)
                board[move] = ' '
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Альфа-бета отсечение
            return min_eval
    
    def get_best_move(self, board):
        """Находит лучший ход для AI с помощью алгоритма Minimax"""
        best_move = -1
        best_value = -float('inf')
        available_moves = [i for i, cell in enumerate(board) if cell == ' ']
        
        # Если AI делает первый ход, выбираем центр или угол для разнообразия
        if len(available_moves) == 9:
            return 4  # Центр - лучший первый ход
        
        # Ищем лучший ход с помощью Minimax
        for move in available_moves:
            board[move] = self.player
            move_value = self.minimax(board, 0, False)
            board[move] = ' '
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move


def get_human_move(board):
    """Получает ход от человека с проверкой ввода"""
    while True:
        try:
            move = input("Введите ваш ход (строка столбец, например '0 1'): ").strip()
            
            # Проверяем, введены ли два числа
            if ' ' in move:
                row, col = map(int, move.split())
                position = row * 3 + col
            else:
                # Или одно число от 0 до 8
                position = int(move)
            
            if position < 0 or position > 8:
                print("Позиция должна быть от 0 до 8. Попробуйте снова.")
            elif board[position] != ' ':
                print("Эта клетка уже занята. Попробуйте снова.")
            else:
                return position
        except ValueError:
            print("Некорректный ввод. Введите два числа через пробел (например '0 1') или одно число от 0 до 8.")


def main():
    """Основная функция игры"""
    game = TicTacToe()
    ai = MinimaxAI(player='O')
    
    print("Добро пожаловать в игру Крестики-нолики с AI!")
    print("Вы играете за 'X', AI играет за 'O'.")
    print("Для хода введите координаты (строка столбец), например '0 1' для первой строки и второго столбца.")
    print("Или введите одно число от 0 до 8, где 0-2 - первая строка, 3-5 - вторая, 6-8 - третья.")
    print("Введите 'q' для выхода из игры.\n")
    
    input("Нажмите Enter для начала игры...")
    
    while True:
        game.print_board()
        
        # Ход человека
        if game.current_player == 'X':
            move = get_human_move(game.board)
            game.make_move(move)
        # Ход AI
        else:
            print("AI думает...")
            ai_move = ai.get_best_move(game.board.copy())
            game.make_move(ai_move)
            print(f"AI сделал ход на позицию {ai_move // 3} {ai_move % 3}")
        
        # Проверяем, закончилась ли игра
        winner = game.check_winner()
        if game.game_over:
            game.print_board()
            if winner == 'Ничья':
                print("Игра окончена! Ничья!")
            else:
                print(f"Игра окончена! Победил {winner}!")
            
            play_again = input("\nХотите сыграть еще раз? (y/n): ").lower()
            if play_again == 'y':
                game.reset_game()
                continue
            else:
                print("Спасибо за игру!")
                break
        
        # Переключаем игрока
        game.switch_player()


if __name__ == "__main__":
    main()