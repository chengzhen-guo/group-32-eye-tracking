from django.http import JsonResponse

def get_pos(reqeust):
	# obtain the current new target (x,y) from the tracking script.  
	# add code
	mock_data = {
		'x': 12,
		'y': 13
	}
	return JsonResponse(mock_data)
