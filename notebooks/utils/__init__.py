import time

__all__ = ['Timer']

class Timer:
	def __init__(self, name: str | None):
		self.name = name

	def __enter__(self):
		self.t0 = time.time()
		return self

	def __exit__(self, type, value, traceback):
		res = ''
		if self.name is not None:
			res += f'[{self.name}] '
		res += f'{(time.time() - self.t0)*1e3} ms'
		print(res)