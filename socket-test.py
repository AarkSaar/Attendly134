with open('names.txt', 'r') as file:
    names = file.read().splitlines()
with open('courses.txt', 'r') as file:
    courses = file.read().splitlines()


async def handle_connection(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        if data['type'] == 'photo':
            try:
                image_data = data['image']
                image = base64_to_opencv_image(image_data)
                cv.imwrite('captured_image.jpg', image)
                
                date, time = extract_date_time(data['date'])
                name = recognise('captured_image.jpg', 'svm_model.pkl')
                
                if name in names:
                    index = names.index(name)
                    sheet = client.open('Your Sheet Name').worksheet(courses[index])
                    row_header = name
                    column_header = date
                    
                    row_cell = sheet.find(row_header)
                    column_cell = sheet.find(column_header)
                    
                    if row_cell and column_cell:
                        row_index = row_cell.row
                        column_index = column_cell.col
                        sheet.update(range_name=f'{column_header}{row_index}', values=['Present'])
                    else:
                        display(f"Row or column header not found: {row_header}, {column_header}")
                else:
                    display(f"Name not found in list: {name}")
            except Exception as e:
                display(f"An error occurred while processing photo_with_info: {e}")

        elif data['type'] == 'photo_with_info':
            try:
                image_data = data['image']
                name = data['name']
                course = data['course']
                
                names.append(name)
                courses.append(course)
                
                sheet = client.open('Your Sheet Name').worksheet(course)
                addName(name, sheet)
                
                image = base64_to_opencv_image(image_data)
                
                existing_folder_path = "dataset"
                new_folder_name = name
                new_folder_path = os.path.join(existing_folder_path, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)
                
                image_path = os.path.join(new_folder_path, f'{name}.jpg')
                cv.imwrite(image_path, image)
                
                faceloader = FACELOADING()
                embedding, label = faceloader.process_single_image(image_path, name)
                
                if embedding is not None:
                    update_npz('face_embeddings.npz', label, embedding)
                    retrain_model('face_embeddings.npz')
                else:
                    display(f"No face detected in image for {name}.")
            
            except Exception as e:
                display(f"An error occurred while processing photo_with_info: {e}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 3000):
        print("WebSocket server started on ws://localhost:8000")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
    