import sys, time, copy, threading, wx

#------------------------------------------------------------------------------
#Library of commonly used functions and classes
#------------------------------------------------------------------------------



#debugging statement print
#acts as print() but inserts caller stack frame information as first argument
#------------------------------------------------------------------------------
def dprint(*args):

	#create fake exception to create a stack trace
	try:
		raise FakeException("Fake Exception")
	except Exception:
		#get second last stack frame (dprint caller)
		frame = sys.exc_info()[2].tb_frame.f_back
	
	#get object of name self in frames function call if exists
	object = frame.f_locals.get("self", None)
	
	#begin string composition
	details = "DEBUG " + time.strftime('%Y/%m/%d %H:%M:%S') + " "
	
	#if function call included self add containing class name
	if object:
		details += object.__class__.__name__ + "::"
	
	#add function name and line number
	details += frame.f_code.co_name + "() line " + str(frame.f_lineno) + ":"
	
	#print details and given parameters
	print(details, *args)



#measures time taken to execute a function
#forwards all given parameters to the function specified in first parameter
#executes given function, returns time taken and functions return value
#------------------------------------------------------------------------------	
def measureTime(function, *args, **kwargs):
	
	timeStart = time.time()
	returnValues = function(*args, **kwargs)
	timeEnd = time.time()
	
	return timeEnd - timeStart, returnValues



#generates a new thread and start executing function passed with arguments
#------------------------------------------------------------------------------	
def newThread(function, *args, **kwargs):
	thread = threading.Thread(target=function, args=(*args,), kwargs=kwargs)
	thread.start()



#creates a wx.App instance and then when called starts the application
#------------------------------------------------------------------------------	
class App():
	def __init__(self):
		self.app = wx.App()

	def show(self):
		self.app.MainLoop()




def readTextFile(filename):

	string = None

	with open(filename,'r') as file:
		string = file.read()

	return string

def readBinaryFile(filename):

	data = bytes()

	with open(filename,'rb') as file:
		string = file.read()
		data = bytes(string)

	return data

#Implement copy and deepcopy functionality into classes
#classes extend to this will be copyable
#------------------------------------------------------------------------------	
class copyable:
	
	#creates a shallow copy of object that extends to this
	#copies object members references
	def copy(self):
		return copy.copy(self)

	#creates a deep copy of object that extends to this
	#creates copies of object members and its object members recursively
	def deepcopy(self):
		return copy.deepcopy(self)
