
class matimer:
	'''number of daughter'''
	nd = 0
	def __init__(self,name,ncall,duration,ratio):
		self.name = name
		self.ncall = ncall
		self.duration = duration
		self.nd=0

	def add_daughter(self, daughter):
		self.daughter.insert(daughter)
		self.nd = nb

