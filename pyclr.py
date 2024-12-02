# pyclr.py


# Цвета в консоли

__version__ = '1.2.1.006 Beta'

"""

Changelog:
	1.2.1
		Добавление Commands — команды консоли
		Рефакторинг до 70% (осталось: ramp, Ramp, Process)
		Перенос класса Process в progressbar.py
	1.2.0
		Добавление функции ramp — не требует создания объекта класса Ramp
		Рефакторинг до 30%
	1.1.0
		Добавление класса Ramp для переливания цветов

"""

import itertools as it

from typing import TypeAlias, Sequence, Iterable, Optional
from threading import Thread as th
from math import sin, cos, pi
from functools import reduce
from time import time
from os import system


# Подготовка и типы
system('') # Включение цветов
Num: TypeAlias = int | float
Coords: TypeAlias = tuple[Num, Num]
RGB: TypeAlias = tuple[Num, Num, Num]
Box: TypeAlias = tuple[Num, Num, Num, Num]

HEX = r'[0-9a-f]{6}'


# Функции
def array_round(x: Num | Iterable[Num]) -> Num | list[Num]:
	""" Округление числа или массива чисел """
	if isinstance(x, Num):
		return round(x)
	return [round(c) for c in x]


def clr(
		fore: Optional[RGB] = None,
		bg: Optional[RGB] = None
	) -> str:
	""" 
	Возвращает строку, содержащую код изменения цвета при её использовании в консоли
	Если не переданы аргументы, очищает цвет
	 * fore — Цвет букв
	 * bg — Цвет фона
	"""
	if fore is None:
		if bg is None:
			return Commands.nocolor
		return Commands.bg.format(*bg)

	if bg is None:
		return Commands.fore.format(*fore)

	return Commands.fore_bg.format(*fore, *bg)


def clean():
	""" Очистить консоль """
	print(end=Commands.clean)


def bright(
		rgb: Sequence[Num], 
		bright: 'True | Num | Sequence[Num]' = True
	) -> list[Num]:
	"""
	Изменение яркости (интенсивности) цвета.
	 * rgb — Исходный цвет
	 * [ bright: True — Умножение яркости (элементов rgb) на x
	   [ bright: Num — Привести к максимальной яркости (до 255)
	   [ bright: Iterable[Num] — Пропорциональное изменение яркости
	"""
	if bright is True: # До 255
		max_value = max(rgb)
		if max_value == 0: # Чёрный цвет
			return (255,)*len(rgb)
		return [round(c*255/max_value) for c in rgb]

	if isinstance(bright, Num):
		return [round(c*bright) for c in rgb] # Домножение

	return [round(rgb[i]*bright[i]) for i in range(len(rgb))] # Пропорциональное домножение

	
def area(
		minrgb: Sequence[Num], 
		rgb: Sequence[Num], 
		maxrgb: Sequence[Num]
	) -> bool: 
	""" Проверяет, попадает ли rgb в область между minrgb и maxrgb включительно """
	for i in range(len(rgb)):
		if not minrgb[i] <= rgb[i] <= maxrgb[i]:
			return False
	return True


def int_to_oneHEX(
		x: int
	) -> str:
	""" Переводит число в два символа HEX """
	if not 0 <= x <= 255:
		raise ValueError("x must be digit in [0; 255]")
	tohex = {
		0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 
		10:'a', 11:'b', 12:'c', 13:'d', 14:'e', 15:'f'}
	return tohex[x // 16] + tohex[x % 16]
	


def RGB_to_HEX(
		rgb: RGB
	) -> HEX:
	""" Переводит RGB в HEX """
	return reduce(lambda a, b: a+b, [int_to_oneHEX(c) for c in rgb])


def ramp(
		ramp: dict[float, Sequence[Num]], 
		val: float, 
		bright: bool = False, 
		smooth: float = 0,
	) -> Sequence[Num]:
	"""
	Плавный переход цветов
	 * ramp — Градиент
	 * val — Значение точки на градиенте для получения цвета
	 * bright — Осветление до 255
	 * smooth — Сглаживание переходов, in [0.0; 1.0]
	"""
	def to_open(t: tuple|Num) -> tuple|Num:
		if len(t) == 1:
			return t[0]
		return t

	ramp = {k: (v,) if isinstance(v, Num) else v for k, v in ramp.items()}
	ks = list(ramp.keys())
	ks.sort()
	if val >= ks[-1]:
		return to_open(ramp[ks[-1]])

	ot = None
	for k in ks:
		if val == k:
			return to_open(ramp[k])
		if val > k:             
			ot = k
		elif val < k:
			do = k
			if ot is None:
				return to_open(ramp[do])
			break

	ramplen: int = len(ramp[ot])
	k: float = (val-ot) / (do-ot) # [0.0-1.0] || [ot-do]
	if smooth: # изменяет k
		#k=k*2-1
		#k=((abs(k)**(1-smooth/2) if k>=0 else -(abs(k)**(1-smooth/2)))+1)/2 #это можно улучшить
		sink = sin(k*pi - pi/2)/2 + 0.5
		k = sink + (k-sink) * (1-smooth)
	rgb = [abs(ramp[ot][i]-(ramp[ot][i]-ramp[do][i])*k) for i in range(ramplen)]

	if bright:
		maxbright = max(ramp[ot])-(max(ramp[ot])-max(ramp[do]))*k-max(rgb)
		maxrgb: int = max(rgb)
		if maxrgb != 0:
			rgb: RGB = [c + maxbright*(c/maxrgb) for c in rgb]
	return to_open(array_round(rgb))



# Классы
class Commands:
	""" Команды изменения цветов в консоли """
	nocolor: str = '\033[0;0m' # Очищение цвета
	fore: str = '\033[38;2;{};{};{}m' # Цвет букв
	bg: str = '\033[48;2;{};{};{}m' # Цвет фона
	fore_bg: str = fore + bg # Цвет букв и фона
	clean: str = '\033[H\033[J' # Очищение консоли (cls)


class Colors:
	""" Цвета в Sublime Text """
	name = clr((216, 222, 233))			# <a> = 1
	num = clr((249, 174, 87))			# a = <1> | class <classname> | (курсор: |)
	func = clr((102, 153, 204))			# <foo>()
	text = clr((153, 199, 148))			# "<text>"
	mark = clr((96, 180, 180))			# <">text" | def <foo>
	oper = clr((198, 149, 198))			# for return def class
	comm = comma = clr((166, 172, 185))	# # ,
	math = clr((249, 123, 87))			# | + - * /
	none = clr((236, 96, 102))          # None


class CmdClr:
	""" Цвета в консоли """
	_PROC =	(200, 50, 0)
	_CMD =	(12, 12, 12)
	SPEED =		clr((200, 15, 30))	# ----- 2.0/5.9 kB<1.1 MB/s>eta 0:00:00 | ERROR
	PROC =		clr(_PROC)			#<---  >2.0/5.9 kB 1.1 MB/s eta 0:00:00
	PROC_END =	clr((200, 155, 0))	#<----->2.0/5.9 kB 1.1 MB/s eta 0:00:00
	TIME =		clr((60, 150, 220))	# ----- 2.0/5.9 kB 1.1 MB/s eta<0:00:00>
	ELEM =		clr((20, 160, 15))	# -----<2.0/5.9 kB>1.1 MB/s eta 0:00:00
	CMD =		clr(_CMD)   # console color


class Ramp:
	def __init__(self, ramp: dict[Num, Num | tuple[Num, ...]], bright_to: Optional[int] = None):
		if all([isinstance(v, Num) for v in ramp.values()]): 
			self.ramp = {k: (v,) for k, v in ramp.items()}
			self.return_rawValue = True
		else: 
			self.ramp = ramp
			self.return_rawValue = False
		
		self.bright_to = bright_to

	def __len__(self):
		return len(self.ramp)

	def __call__(self, value: Num) -> RGB:
		keys = sorted(self.ramp.keys()) # сортировка по возрастанию

		if len(keys) == 1:
			res = self.ramp[keys[0]]

		# если значение за границами ramp
		elif value < keys[0]:
			res = self.ramp[keys[0]]
		elif value > keys[-1]:
			res = self.ramp[keys[-1]]

		else:
			# если значение внутри ramp
			for range_keys in zip(keys, keys[1:]):
				if value in range_keys:
					res = self.ramp[range_keys[0] if value == range_keys[0] else range_keys[1]]
				if range_keys[0] < value < range_keys[1]:
					k = (value-range_keys[0])/(range_keys[1]-range_keys[0]) # % движения от начала до конца отрезка
					res = [c1+(c2-c1)*k for c1, c2 in zip(*[self.ramp[key] for key in range_keys])]

		if not self.bright_to is None:
			k = max(res)/self.bright_to
			res = [c/k for c in res] if k else (self.bright_to,)*3
		return res[0] if self.return_rawValue else tuple([min(round(c), 255) for c in res])

	def __str__(self):
		return f'Ramp(ramp={self.ramp}' + ('' if self.bright_to is None else f', bright_to={self.bright_to}') + ')'


'''
class Process:
	""" шкала обработанных элементов """
	def __init__(self, total_n: Num, min_step: Num = 0):
		self.total_n = total_n
		self.min_step = min_step

		self.elements: int = 40
		self.last_n: Num = 0
		self.start_time: Num = time()
		self.ended: bool = False

		self.segments: tuple[str] = tuple('-–—')
		self.segments_len: int = len(self.segments)
		
		self.to_num: Callable[[Num], Num] = lambda n: round(n, 1) if isinstance(n, float) else n

	def get_strtime(self, sec: Num) -> str:
		""" время строкой """
		sec_to_str: Callable[[Num], str] = lambda t: str(int(t)) if t >= 10 else f'0{int(t)}'
		return f'{int(sec//3600)}:{sec_to_str(sec//60)}:{sec_to_str(sec%60)}'

	def print(self, n: Num, label: str = ''):
		if self.ended:
			return
		if n - self.last_n < self.min_step and n != self.total_n: # слишком мелкий шаг
			return
		self.last_n: Num = n

		label: str = f'{CmdClr.PROC_END}{label}: ' if label else ''

		ended_time: float = time() - self.start_time
		percents: float = n / self.total_n
	
		if percents < 0.0001 or ended_time < 0.0001:
			# начало загрузки
			print(end = \
				f'\t{label}' + ' '*self.elements +
				f' {CmdClr.ELEM}0/{self.to_num(self.total_n)} el'
				f' {CmdClr.TIME}0%'
				f' {clr()}eta {CmdClr.TIME}{self.get_strtime(0)}'
				f'{clr()}\r'
			)
	
		elif 1-percents < 0.001:
			# загружено
			print(end = \
				f'\t{label}{CmdClr.PROC_END}' + '—'*self.elements +
				f' {CmdClr.ELEM}{self.to_num(self.total_n)}/{self.to_num(self.total_n)} el'
				f' {CmdClr.SPEED}{round(n / ended_time, 1)} el/s'
				f' {CmdClr.TIME}{n / self.total_n * 100:.4f}%'
				f' {clr()}eta {CmdClr.TIME}{self.get_strtime(max(0, int(ended_time/percents - ended_time)))}'
				f'\t{clr()}\n\n'
			)
			self.ended: bool = True
	
		else:
			# процесс
			completed_el: float = self.elements*percents
			print(end = \
				f'\t{label}{CmdClr.PROC}' + '—'*int(completed_el) + 
					self.segments[int(completed_el % 1 * self.segments_len)] +
					' '*int(self.elements-completed_el) +
				f' {CmdClr.ELEM}{self.to_num(n)}/{self.to_num(self.total_n)} el'
				f' {CmdClr.SPEED}{round(n / ended_time, 1)} el/s'
				f' {CmdClr.TIME}{n / self.total_n * 100:.4f}%'
				f' {clr()}eta {CmdClr.TIME}{self.get_strtime(max(0, int(ended_time/percents - ended_time)))}'
				f' {clr()}\r'
			)
'''


def main():

	# Clr
	print(clr((0, 255, 200)) + 'Hello world!' + clr() + '\n')

	# Ramp
	text: str = 'Hello Ramp!'
	r: Ramp = Ramp({0: (255, 0, 200), len(text)-1: (255, 200, 0)})

	for i in range(len(text)):
		print(end = clr(r(i)) + text[i])
	print(clr()) # Сброс цвета


if __name__ == '__main__':
	main()
