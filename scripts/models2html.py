import sys

def main(fname):
	row_format=' '\
	'<tr>'\
	'		 <td width="25%">Trainable params: {params}<br>Time stamp: {identifier}<br><img width="100%" src="{identifier}_model.png">'\
	'	</td><td width="25%"><img width="100%" src="{identifier}_loss.png">'\
	'	</td><td width="25%"><img width="100%" src="{identifier}_acc.png">'\
	'	</td><td width="25%"><iframe src="{identifier}_hyperparameters.json"></iframe>'\
	'	</td>'\
	'</tr>'\
	' '
	models = []
	with open(fname, 'r') as f:
		for l in f:
			(identifier, params) = l.strip().split('|')
			params = float(params.replace(',', ''))
			models.append( (params, identifier) )
	models.sort()
	print """<table style="width:100%">"""
	for m in models:
		print row_format.format(params=m[0], identifier=m[1])
	print """</table>"""

main(sys.argv[1])
