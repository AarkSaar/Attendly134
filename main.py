from utils import *

## Authenticating google sheets api
# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Authenticate using the JSON key file
credentials = Credentials.from_service_account_file("proj-sheets.json", scopes=scope)

# Authenticate with gspread
client = gspread.authorize(credentials)

# Open the Google Sheet by name
spreadsheet = client.open("Attendly-scan")
sheet = spreadsheet.worksheet('Course info')

keys = sheet.col_values(1)
values = sheet.col_values(2)

course_times = {k:v for k, v in zip(keys, values) if k and v}
# print(course_times)
with open('names.txt', 'r') as file:
    names = file.read().splitlines()
with open('courses.txt', 'r') as file:
    courses = file.read().splitlines()


async def handle_connection(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        print(f"Received data")  # Add this line for debugging
        
        if data['type'] == 'photo':
            print(1)
            image_data = data['image']
            image = base64_to_opencv_image(image_data)
            cv.imwrite('captured_image.jpg', image)
            print(2, data['date'])
            date, time = extract_date_time(data['date'])
            date = str(date)
            print(3, date, type(date))
            faceloader = FACELOADING()
            X = faceloader.process_image('captured_image.jpg')
            EMBEDDED_X = []
            for img in X:
                embedding = get_embedding(img)
                EMBEDDED_X.append(embedding)
            EMBEDDED_X = np.asarray(EMBEDDED_X)
            name = recognise('model.pkl', 'faces_embeddings.npz', EMBEDDED_X)
            print(f"Recognised name: {name}")  # Add this line for debugging
            print(4, name, type(name))
            
            if name in names:
                index = names.index(name)
                sheet = spreadsheet.worksheet(courses[index])
                row_header = name
                column_header = get_column_header(sheet, date)
                
                row_cell = sheet.find(row_header)
                row_index = row_cell.row
                print(5, column_header, row_index)
                format = "%H:%M:%S"
                late_time = course_times[courses[index]]
                if datetime.strptime(time, format) - datetime.strptime(late_time, format) < timedelta(minutes=30):
                    sheet.update([['Present']], f'{column_header}{row_index}')
                else:
                    sheet.update([['Late']], f'{column_header}{row_index}')
                print("Attendance logged")
                await websocket.send(json.dumps({"message": f"{name} has logged in for {courses[index]} today"}))
                
            else:
                print(f"Name not found in list: {name}")  # Changed from display to print
                await websocket.send(json.dumps({"Name not found or Face not Registered"}))

        elif data['type'] == 'photo_with_info':
            image_data = data['image']
            name = data['name']
            course = data['course']
            print(1)
            names.append(name)
            courses.append(course)
            print(2)
            sheet = client.open('Attendly-scan').worksheet(course)
            addName(name, sheet)
            print(3)
            image = base64_to_opencv_image(image_data)
            print(4)
            existing_folder_path = "dataset"
            new_folder_name = name
            new_folder_path = os.path.join(existing_folder_path, new_folder_name)
            os.makedirs(new_folder_path)
            print(5)
            image_path = os.path.join(new_folder_path, f'{name}.jpg')
            cv.imwrite(image_path, image)
            print(6)
            faceloader = FACELOADING()
            X = faceloader.process_image(f'dataset/{name}/{name}.jpg')
            print(7)
            EMBEDDED_X = []
            for img in X:
                embedding = get_embedding(img)
                EMBEDDED_X.append(embedding)
            EMBEDDED_X = np.asarray(EMBEDDED_X)
            name = np.array([name])
            print(8, EMBEDDED_X.shape, name.shape)
            update_npz("faces_embeddings.npz", EMBEDDED_X, name)
            print(9)
            retrain_model('faces_embeddings.npz', 'model.pkl')
            print(10)
            with open('names.txt', 'w') as file:
                for i in names:
                    file.write(f"{i}\n")
            with open('courses.txt', 'w') as file:
                for i in courses:                                                                                                                                                                                                                    
                    file.write(f"{i}\n")
            print("11. Done")
            #await websocket.send(json.dumps({"message": f"{name} has been registered to {name[course]}"}))
        
        
async def start_server():
    server = await websockets.serve(handle_connection, "localhost", 8080)
    print("WebSocket server started on ws://localhost:8080")
    try:
        # Keep the server running
        await asyncio.Future()  # Run forever
    finally:
        # Ensure the server is closed on exit
        server.close()
        await server.wait_closed()

# Run the server
asyncio.run(start_server())
