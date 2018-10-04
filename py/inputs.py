import wx
from common import *

class Input:
	def __init__(self, name, value, onUpdate):
		self.name = name
		self.value = value
		self.update = onUpdate
		self.state = None
		self.view = None

	def setValue(self, value):
		self.value = value
		if self.view != None:
			self.updateWidget()
			self.processUpdate(None)

	def updateWidget(self):
		pass

	def processUpdate(self, event):
		pass

	def onUpdate(self):
		if self.update != None:
			self.update(self, self.value, self.state)

	def createWxWidget(self):
		pass

class InputButton(Input):
	def __init__(self, name, onClick=None):
		Input.__init__(self, name, None, onClick)

	def updateWidget(self):
		self.view.SetValue(self.name)

	def processUpdate(self, event):
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.view = wx.Button(panel, label=self.name)
		self.view.Bind(wx.EVT_BUTTON, self.processUpdate)

		return self.view

class InputSlider(Input):
	def __init__(self, name, value, onUpdate=None, sMin=0, sMax=100, sSteps=1):
		Input.__init__(self, name, value, onUpdate)

		self.min = sMin
		self.max = sMax
		self.steps = sSteps
		self.valueView = None

	def updateWidget(self):
		self.view.SetValue(self.value)

	def processUpdate(self, event):
		self.value = (self.view.GetValue() - self.min) // self.steps * self.steps + self.min
		self.valueView.SetLabelText(str(self.value))
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.valueView = wx.StaticText(panel, label=str(self.value)+"    ")
		self.view = wx.Slider(panel, value=self.value, minValue=self.min, maxValue=self.max)
		self.view.Bind(wx.EVT_SCROLL, self.processUpdate)

		self.view.SetMinSize((300, 30))

		sizer = wx.BoxSizer(wx.HORIZONTAL)
		sizer.Add(self.view)
		sizer.Add(self.valueView, wx.EXPAND)

		return sizer

class InputCheckBox(Input):
	def __init__(self, name, value, onUpdate=None):
		Input.__init__(self, name, value, onUpdate)

	def updateWidget(self):
		self.view.SetValue(self.value)

	def processUpdate(self, event):
		self.value = self.view.GetValue()
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.view = wx.CheckBox(panel)
		self.view.SetValue(self.value)
		self.view.Bind(wx.EVT_CHECKBOX, self.processUpdate)

		return self.view

class InputText(Input):
	def __init__(self, name, value, onUpdate=None):
		Input.__init__(self, name, value, onUpdate)

	def updateWidget(self):
		self.view.SetValue(self.value)

	def processUpdate(self, event):
		self.value = self.view.GetValue()
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.view = wx.TextCtrl(panel, value=self.value)
		self.view.Bind(wx.EVT_TEXT, self.processUpdate)

		return self.view

class InputColor(Input):
	def __init__(self, name, value, onUpdate=None):
		Input.__init__(self, name, value, onUpdate)

	def updateWidget(self):
		self.view.SetColour(self.value)

	def processUpdate(self, event):
		color = self.view.GetColour()
		self.value = (color.Red(), color.Green(), color.Blue(), color.Alpha())
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.view = wx.ColourPickerCtrl(panel, colour=self.value)
		self.view.Bind(wx.EVT_COLOURPICKER_CHANGED, self.processUpdate)

		return self.view

class InputFile(Input):
	def __init__(self, name, value, onUpdate=None):
		Input.__init__(self, name, value, onUpdate)

	def updateWidget(self):
		self.view.SetPath(self.value)

	def processUpdate(self, event):
		self.value = self.view.GetPath()
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.view = wx.FilePickerCtrl(panel, path=self.value)
		self.view.Bind(wx.EVT_FILEPICKER_CHANGED, self.processUpdate)

		return self.view

class InputFolder(Input):
	def __init__(self, name, value, onUpdate=None):
		Input.__init__(self, name, value, onUpdate)

	def updateWidget(self):
		self.view.SetPath(self.value)

	def processUpdate(self, event):
		self.value = self.view.GetPath()
		self.onUpdate()

	def createWxWidget(self, panel):
		
		self.view = wx.DirPickerCtrl(panel, path=self.value)
		self.view.Bind(wx.EVT_DIRPICKER_CHANGED, self.processUpdate)

		return self.view

class InputFrame(wx.Frame):
	def __init__(self, parent, title, inputs):
		wx.Frame.__init__(self, None, title=title)

		self.panel = wx.Panel(self)
		self.sizer = wx.GridBagSizer(hgap=5, vgap=5)

		row = 0
		for control in inputs:
			txt = wx.StaticText(self.panel, label = control.name)
			view = control.createWxWidget(self.panel)
			view.SetMinSize((300, 0))

			self.sizer.Add(txt, pos = (row, 0), flag = wx.EXPAND)
			self.sizer.Add(view, pos = (row, 1), flag = wx.EXPAND)

			row += 1

		self.panel.SetSizerAndFit(self.sizer)
		self.Fit()
		self.Show()

class Inputs:
	def __init__(self, title="Inputs", state=None):
		self.inputs = {}
		self.title = title
		self.state = state

	def addInput(self, controls, window=None):
		controls.state = self.state

		if window == None:
			window = self.title

		if window in self.inputs:
			self.inputs[window].append(controls)
		else:
			self.inputs[window] = [controls]

	def getFrames(self):
		frames = []
		
		for window in self.inputs:
			frames.append(InputFrame(self, window, self.inputs[window]))

		return frames

	def show(self):
		
		self.app = wx.App()
		self.getFrames()
		self.app.MainLoop()

def exampleUpdate(control, value, state):
	dprint(control.name, value)

def exampleUpdateSlider(control, value, state):
	dprint(control.name, value)

	state[0].setValue(value)

if __name__ == "__main__":

	button1 = InputButton("Button", onClick=exampleUpdate)
	slider1 = InputSlider("Slider A", 10, onUpdate=exampleUpdate)
	slider2 = InputSlider("Slider B", 10, onUpdate=exampleUpdateSlider)
	slider3 = InputSlider("Slider C", 10, onUpdate=exampleUpdateSlider)
	checkbox1 = InputCheckBox("Important", True, onUpdate=exampleUpdate)
	color1 = InputColor("Color", "#0088FF", onUpdate=exampleUpdate)
	color2 = InputColor("Color", (255, 128, 0, 255), onUpdate=exampleUpdate)
	text1 = InputText("Text", "Text", onUpdate=exampleUpdate)
	file1 = InputFile("File", "Text", onUpdate=exampleUpdate)
	folder1 = InputFolder("Folder", "Text", onUpdate=exampleUpdate)

	inputs = Inputs(state=[slider1])

	inputs.addInput(button1, "Inputs")
	inputs.addInput(slider1, "Inputs")
	inputs.addInput(slider2, "Inputs")
	inputs.addInput(slider3, "Inputs")
	inputs.addInput(checkbox1, "Inputs")
	inputs.addInput(color1, "Inputs")
	inputs.addInput(color2, "Inputs")
	inputs.addInput(text1, "Inputs")
	inputs.addInput(file1, "Inputs")
	inputs.addInput(folder1, "Inputs")

	inputs.show()