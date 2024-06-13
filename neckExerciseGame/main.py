import cv2
import dlib
import numpy as np
import random
import time

class Maze:
    def __init__(self):
        self.width = 400
        self.height = 400
        self.walls = [
            ((50, 50), (350, 350)),
            ((50, 350), (350, 50)),
            ((150, 0), (150, 250)),
            ((250, 400), (250, 150)),
            ((0, 150), (150, 150)),
            ((250, 250), (400, 250))
        ]
        self.maze = self.create_maze()

    def create_maze(self):
        maze = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for wall in self.walls:
            cv2.line(maze, wall[0], wall[1], (255, 255, 255), 2)
        return maze

    def check_collision(self, position):
        if position[0] < 0 or position[0] >= self.width or position[1] < 0 or position[1] >= self.height:
            return True
        if self.maze[position[1], position[0]].all() == 255:
            return True
        return False

    def find_random_position(self):
        while True:
            pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            if not self.check_collision(pos):
                return pos

class Player:
    def __init__(self, maze):
        self.maze = maze
        self.radius = 10
        self.position = self.maze.find_random_position()

    def move(self, nose_tip):
        new_pos = [np.clip(nose_tip[0], 0, self.maze.width - 1), np.clip(nose_tip[1], 0, self.maze.height - 1)]
        if not self.maze.check_collision(new_pos):
            self.position = new_pos

    def draw(self, frame):
        cv2.circle(frame, tuple(self.position), self.radius, (0, 255, 0), -1)

class Game:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.maze = Maze()
        self.player = Player(self.maze)
        self.end_point = self.maze.find_random_position()
        self.end_radius = 20
        self.start_time = time.time()
        self.game_over = False

    def detect_nose_tip(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            return nose_tip
        return None

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            if not self.game_over:
                nose_tip = self.detect_nose_tip(frame)
                if nose_tip:
                    self.player.move(nose_tip)
                    cv2.circle(frame, nose_tip, 5, (0, 255, 0), -1)
                    
                maze_with_player = self.maze.maze.copy()
                self.player.draw(maze_with_player)

                cv2.circle(maze_with_player, self.end_point, self.end_radius, (0, 0, 255), -1)

                if np.linalg.norm(np.array(self.player.position) - np.array(self.end_point)) < (self.player.radius + self.end_radius):
                    self.game_over = True
                    elapsed_time = time.time() - self.start_time
                    cv2.putText(maze_with_player, f"You Win! Time: {elapsed_time:.2f}s", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Maze", maze_with_player)
                cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = Game()
    game.run()
