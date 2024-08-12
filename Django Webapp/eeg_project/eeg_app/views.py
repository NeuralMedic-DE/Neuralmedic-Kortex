from django.shortcuts import render
from django.http import HttpResponse  # Import HttpResponse
import os
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
from django.http import JsonResponse
import requests
from django.views.decorators.csrf import csrf_exempt
import yaml

def index(request):
    return render(request, 'eeg_app/index.html')

def download_file(request):
    # Path to the file you want to serve
    file_path = '/home/public/EEG-RTV/eeg_project/eeg_app/static/eeg_app/files/eeg_recording.txt'  # Change this to the path of your file

    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/octet-stream")
            response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
            return response
    else:
        return HttpResponse("File not found.", status=404)


@csrf_exempt
def send_metadata(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            yaml_data = {
                "eeg_setup": {
                    "recording": {
                        "sample_rate": data.get("sampleRate"),
                    },
                    "electrodes": {
                        "type": data.get("electrodeType"),
                        "impedance": data.get("electrodeImpedance"),
                    },
                    "channels": {
                        "number_of_channels": data.get("numberOfChannels"),
                        "names": data.get("channelNames"),
                        "type": data.get("channelType"),
                        "frequency_range": data.get("frequencyRange"),
                        "filter": data.get("filter"),
                        "units": data.get("units"),
                    },
                    "hardware_setup": {
                        "amplifier": data.get("amplifier"),
                    },
                    "environment": {
                        "room_temperature": data.get("roomTemperature"),
                        "lighting_conditions": data.get("lightingConditions"),
                    },
                    "comments": data.get("comments"),
                },
                "subject": {
                    "demographics": {
                        "id": data.get("subjectId"),
                        "age": data.get("subjectAge"),
                        "gender": data.get("subjectGender"),
                    },
                    "existing_issues": {
                        "medical_conditions": data.get("subjectMedicalConditions", "").split(","),
                    },
                },
                "window_details": {
                    "duration": data.get("duration"),
                },
                "additional_notes": data.get("additionalNotes"),
            }

            yaml_output = yaml.dump(yaml_data)

            webhook_url = 'https://webhook.site/521e973b-eba5-42d6-ab11-2c40312ccd45'
            response = requests.post(webhook_url, json={"yaml": yaml_output})
            response.raise_for_status()

            try:
                response_data = response.json()
            except ValueError:
                response_data = response.text

            return JsonResponse({'status': 'success', 'data': response_data, 'yaml': yaml_output})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'invalid_method'}, status=405)
