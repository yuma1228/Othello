import sys
import pygame
from pygame.locals import *
import math
from SizedOthello import SizedOthello


CELL = 64
MARGIN = 44


class GUI_SizedOthello(SizedOthello):
    def __init__(self, size=6):
        super().__init__(size)

    def play(self, black=None, white=None):
        pygame.init()
        board_px = CELL * self.size + MARGIN * 2
        screen = pygame.display.set_mode((board_px, board_px))
        pygame.display.set_caption(f"Othello {self.size}x{self.size}")
        color = 1
        pos_puts = self.possible_puts(1)
        winner = -1
        end_flag = False

        while True:
            player = black if color == 1 else white
            screen.fill((0, 155, 0))

            # 一手前の手をハイライト
            if len(self.history) > 0 and self.history[-1] is not None:
                x, y = self.history[-1]
                pygame.draw.rect(
                    screen, (180, 180, 0),
                    Rect(MARGIN + CELL * x, MARGIN + CELL * y, CELL, CELL),
                )

            # 枠線
            for i in range(self.size + 1):
                pygame.draw.line(screen, (0, 0, 0),
                    (i * CELL + MARGIN, MARGIN),
                    (i * CELL + MARGIN, CELL * self.size + MARGIN), 1)
            for i in range(self.size + 1):
                pygame.draw.line(screen, (0, 0, 0),
                    (MARGIN, i * CELL + MARGIN),
                    (CELL * self.size + MARGIN, i * CELL + MARGIN), 1)

            # 石
            for y in range(self.size):
                for x in range(self.size):
                    s = self.board[y][x]
                    if s == 0:
                        continue
                    c = (0, 0, 0) if s == 1 else (255, 255, 255)
                    cx = MARGIN + CELL // 2 + x * CELL
                    cy = MARGIN + CELL // 2 + y * CELL
                    pygame.draw.circle(screen, c, (cx, cy), 25)

            # 置ける場所を薄く表示
            for px, py in pos_puts:
                if player is None:
                    cx = MARGIN + CELL // 2 + px * CELL
                    cy = MARGIN + CELL // 2 + py * CELL
                    pygame.draw.circle(screen, (0, 100, 0), (cx, cy), 8)

            pygame.display.update()

            # AIの手番
            if player is not None:
                x, y = player(self, color)
                self.put(x, y, color)
                self.history.append((x, y))
                color = 3 - color
                if not self.is_possible_to_put_anywhere(color):
                    color = 3 - color
                w = self.winner()
                if w is not None:
                    winner = w
                    break
                pos_puts = self.possible_puts(color)
                pygame.display.update()
                continue

            # 人間の手番
            for e in pygame.event.get():
                if e.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif e.type == MOUSEBUTTONDOWN:
                    x, y = map(lambda v: math.floor((v - MARGIN) / CELL), e.pos)
                    if (x, y) in pos_puts:
                        self.put(x, y, color)
                        self.history.append((x, y))
                        color = 3 - color
                        if not self.is_possible_to_put_anywhere(color):
                            color = 3 - color
                        w = self.winner()
                        if w is not None:
                            winner = w
                            end_flag = True
                        pos_puts = self.possible_puts(color)
                elif e.type == MOUSEMOTION:
                    x, y = map(lambda v: math.floor((v - MARGIN) / CELL), e.pos)
                    if (x, y) in pos_puts:
                        pygame.mouse.set_cursor(pygame.cursors.diamond)
                    else:
                        pygame.mouse.set_cursor(pygame.cursors.arrow)

            if end_flag:
                break
            pygame.display.update()

        # 結果表示
        screen.fill((0, 155, 0))
        font = pygame.font.SysFont(None, 48)
        msg = {0: "Draw!", 1: "Black wins!", 2: "White wins!"}.get(winner, "")
        text = font.render(msg, True, (255, 255, 255))
        rect = text.get_rect(center=(board_px // 2, board_px // 2))
        screen.blit(text, rect)
        pygame.display.update()

        # クリックで閉じる
        waiting = True
        while waiting:
            for e in pygame.event.get():
                if e.type == QUIT or e.type == MOUSEBUTTONDOWN:
                    waiting = False
        pygame.quit()
        return winner
