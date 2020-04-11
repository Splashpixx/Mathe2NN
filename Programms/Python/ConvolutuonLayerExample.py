x = [[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]]

def scale(x):
	n = 3
	that = [[[0 for k in range(9)] for j in range(n)] for i in range(n)]
	row = 0
	col = 0
	
	
	while col < 3:
		row = 0
		while row < 3:
			y = 0
			
			for i in range(n):
				for j in range(n):
					
					that[col][row][y] = x[i + col][j + row]
					y  = y+1
				
			row = row + 1
		col = col +1
					
	return that


print("Das Image ist :",x)
				
def knumb(z):
	k = [1,0,1,0,1,0,1,0,1]
	w = k[z]
	return w

def convNet(x):
	out = [[0 for q in range(3)] for p in range(3)]
	l = scale(x)
	p = 0
	q = 0
	for a in range(len(l)):
		for b in range(len(l[a])):
			z = 0
			erg  = 0
			for c in range(len(l[a][b])):
				calc = 0
				
				pos1 = knumb(z)
				pos2 = l[a][b][c]
				calc = (pos1 * pos2)
				
				erg = erg + calc
				z = z + 1
				
			out[p][q] = erg

			if q == 2:
				p = p + 1
				q = 0
			else:
				q = q  + 1
			
	return out
print("")
print("Das Convolved feature ist:",convNet(x))
