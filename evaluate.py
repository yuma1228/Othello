from othello import Othello

def generate_evaluation_table(individual: list, offset: int) -> list[list[int]]:
    table = [[0] * 8 for _ in range(8)]
    for i in range(4):
        for j in range(4):
            table[i][j] = individual[offset + i * 4 + j]  # 左上
            table[i + 4][j] = individual[offset + (3 - i) * 4 + j]  # 左下
            table[i][j + 4] = individual[offset + 4 * (i + 1) - (j + 1)]  # 右上
            table[i + 4][j + 4] = individual[offset + (3 - i) * 4 + (3 - j)]  # 右下
    return table

def input_table(individual: list) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    return (
        generate_evaluation_table(individual, 0),   
        generate_evaluation_table(individual, 16),  
        generate_evaluation_table(individual, 32)   
    )

def evaluate_board_ngn(board:list[list[int]], color:int)-> int : 
    #individual = [76.32979597875611, 155.02097502448746, 80.21479588730273, -57.11627639363957, 6.368691011774339, 107.90361876092376, -43.315618467197325, 67.37930950146608, -157.71874531078652, -78.04039652208158, -48.95180234112112, 4.28172029886239, -29.25181561254189, -16.47893824725932, 127.23177073845451, -37.93457843823043, -46.26370399397668, -68.35245739870845, -29.36601427006844, -97.62578000985552, 110.69233168511644, -23.98252588862141, -87.95760213953896, 35.1163050337853, -95.68104953627522, -86.38719713713654, -8.181796888333814, -82.23446455537324, -57.58377900135889, -121.28657297804868, 13.70639899168555, 93.00965595761866, 20.236790360342006, 151.3537744955126, 18.595252743216836, -51.97702238189073, 10.770162885042808, 2.0465457203132997, -68.26653115437024, 7.196200893902509, -106.80396974437706, 52.30814625201634, 27.21809030731823, -2.927907751501147, -118.98175181328763, 54.80004027378612, -198.5173970311548, -45.46883001943995, -187.20890983990466, -146.70306121112083]
    #individual = [100,10,40,30,10,5,8,6,40,8,15,10,30,6,10,10,100,10,40,30,10,5,8,6,40,8,15,10,30,6,10,10,100,10,40,30,10,5,8,6,40,8,15,10,30,6,10,10,-100,50]
    #individual = [-94.81412133041688, 42.34619796616179, -71.78233287428331, -85.47794588988042, 33.338447206385524, -85.89216752588028, -39, -85.75938983758539, -74, -85.35814081651553, 75.05635106267562, 85, 30, -81, -29.201291396967697, 6, -23.008473924333916, 34.43190441532173, -58, 54, -52.04411832664049, 46.025282385522964, 6.553826368084452, 88.31648212544917, 26.680405750692366, 20.105926783752462, 58, 11, -76, 27.618075371748525, -20.816383250996733, -56.70080176640013, 47.53832016883305, -96.91658885206847, 73.96204530097711, 75.00448645489318, -5.627017726758881, -67.78797799133186, 20.862673866327846, -44.73083390444462, 98.03037395038163, 92.5391355113002, 73, -92.48080171327389, -75.22240867045184, 6.814106499641621, -79.10526518261993, -41.073090059246084, -69.59738091935485, 49.120970917809665]
    individual = [106.78749765600816, 72.12713577258272, 31.538176426160092, 16.875786501434398, 40.37503000945599, -1.1692915942502653, 108.7210201402932, 26.254654819187255, 16.09347083399672, 12.809653205209763, 6.09748464021267, -66.95959241482552, 123.89020090179316, -7.692764999154349, -127.93071827337522, 38.44653912800292, 91.95471010740121, 80.98252384903839, -90.65063996187627, -65.40926890710882, 16.68858676952031, 107.97850534488391, 55.627215672266175, -52.80159375555259, -30.520747040240668, -12.576114822985428, 25.50631234181281, -32.84986397972088, -120.61049049799205, -28.678466847278507, 60.52652531132287, -41.46937394534629, 21.46997297615289, -44.45735812955353, -88.66904685812867, -37.69687508957992, 88.66238924080079, -79.67494757730736, 13.323519844641194, 31.70126605533508, 77.01531838015535, 92.77599888213776, -28.78668522566035, -10.093961014143542, 97.35947814752322, 26.12142277806955, -61.81051846135712, 74.01301197046749, -45.068907537150125, -76.89700028134232]
    evaluation_table, evaluation_table2, evaluation_table3 = input_table(individual[:48])

    n = sum(1 for row in board for cell in row if cell != 0)

    oth = Othello()
    oth.board = board
    if oth.winner() == color:
        return 1000 * (sum(1 if board[i][j] == 1 else -1 for i in range(8) for j in range(8))) * (1 if color == 1 else -1)

    # 評価値の計算
    score = 0
    opponent = 3 - color
    side_continous = individual[49]
    move_eval = individual[48]

    def update_score(evaluation_table: list[list[int]]):
        nonlocal score
        for y in range(8):
            for x in range(8):
                if board[y][x] == color:
                    score += evaluation_table[y][x]
                elif board[y][x] == opponent:
                    score -= evaluation_table[y][x]
                # 端の連続配置評価
                if (y in {0, 7} and 1 <= x <= 6 and board[y][x] == color and 
                   (board[y][x - 1] == opponent or board[y][x + 1] == opponent)) or \
                   (x in {0, 7} and 1 <= y <= 6 and board[y][x] == color and 
                   (board[y - 1][x] == opponent or board[y + 1][x] == opponent)):
                    score += side_continous


    if n <= 10:
        update_score(evaluation_table)
    elif n <= 40:
        update_score(evaluation_table2)
    else:
        update_score(evaluation_table3)
        score -= len(oth.possible_puts(opponent)) * move_eval

    return score if color == 1 else -score


def evaluate_board_nkmr(board: list[list[int]], color:int) -> int:
    result = 0
    n = 0  # 置かれた石の数
    for y in range(8):
        for x in range(8):
            if board[y][x] != 0:
                n += 1
    oth=Othello()
    oth.board=board
    if oth.winner()==color:
        black=0
        for i in range(8):
            for j in range(8):
                if board[i][j]==1:black+=1
                elif board[i][j]==2:black-=1
        return (black*1000)*(1 if color==1 else -1)
    #相手のおける場所が少ないほうが良い
    result=-len(oth.possible_puts(3-color))*5
    # 隅の列全部とったら大加点
    #TODO 確定石は+40点

    for c in range(1, 3):
        e = 0
        if board[0].count(c) == 8:
            e += 100
        if board[7].count(c) == 8:
            e += 100
        c0 = 0
        c7 = 0
        for y in range(8):
            if board[y][0] == c:
                c0 += 1
            if board[y][7] == c:
                c7 += 1
        if c0 == 8:
            e += 100
        if c7 == 8:
            e += 100
        result += e if c == 1 else -e
    
    for y in range(8):
        for x in range(8):
            if board[y][x] == 0:
                continue
            e = 1

            # 序盤は数いらない
            if n < 20:
                e -= 3
            #終盤は数優先
            if n>40:
                e+=15
            if n>=60:
                e+=45

            if (
                (x, y) == (0, 0)
                or (x, y) == (0, 7)
                or (x, y) == (7, 0)
                or (x, y) == (7, 7)
            ):
                e += 200
            
            #隅は加点
            if (x == 0 or x == 7 or y == 0 or y == 7):
                e += 15
            # とってない角の隣はペナルティ
            if board[y][x]==1:
                if board[0][0] != 1 and ((x, y) == (1, 1) or (x, y) == (0, 1) or (x, y) == (1, 0)):
                    e-=60
                if board[0][7] != 1 and ((x, y) == (1, 6) or (x, y) == (1, 7) or (x, y) == (0, 6)):
                    e-=60
                if board[7][0] != 1 and ((x, y) == (6, 1) or (x, y) == (6, 0) or (x, y) == (7, 1)):
                    e-=60
                if board[7][7] != 1 and ((x, y) == (6, 7) or (x, y) == (6, 6) or (x, y) == (7, 6)):
                    e-=60
                
            if board[y][x]==2:
                if board[0][0] != 2 and ((x, y) == (1, 1) or (x, y) == (0, 1) or (x, y) == (1, 0)):
                    e-=60
                if board[0][7] != 2 and ((x, y) == (1, 6) or (x, y) == (1, 7) or (x, y) == (0, 6)):
                    e-=60
                if board[7][0] != 2 and ((x, y) == (6, 1) or (x, y) == (6, 0) or (x, y) == (7, 1)):
                    e-=60
                if board[7][7] != 2 and ((x, y) == (6, 7) or (x, y) == (6, 6) or (x, y) == (7, 6)):
                    e-=60
                
           

            if x ==0 or x==2 or x==5 or x==7:
                e += 1
            else:
                e -= 1
            if y==0 or y==2 or y==5 or y==7:
                e += 1
            else:
                e -= 1

            result += e if board[y][x] == 1 else -e
    return result if color == 1 else -result