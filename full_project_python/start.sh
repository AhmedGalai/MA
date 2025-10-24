# terminal 1
uvicorn mock_pose_api:app --host 0.0.0.0 --port 9000 --reload
# terminal 2
uvicorn mock_api:app --host 0.0.0.0 --port 8000 --reload

py tk_client.py
