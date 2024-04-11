import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from time import time
from util import util
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# python backend_api.py --dataroot / --name RTX-3080-FL-FINAL-continue-fl10 --model test --no_dropout --eval --load_iter 3
# images_dir = os.path.join("/media/ramindu/90F29ACFF29AB8C4/Project/fed-cycle-gan-renew/public/userImages/")
# images_dir = os.path.join("/media/ramindu/90F29ACFF29AB8C4/Project/fedCycleGan/public/userImages/")
images_dir = os.path.join("/media/ramindu/90F29ACFF29AB8C4/Project/fedCycleGan/userImages/")
if not os.path.isdir(images_dir):
    os.makedirs(images_dir)

app = FastAPI()

# ---- Model Testing ----
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.dataroot = images_dir
opt.num_threads = 0  # test code only supports num_threads = 0
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
opt.preprocess = "none"
model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)  # regular setup: load and print networks; create schedulers
eval_metrics_avg = {}  # initialize average metrics
test_len = 0
model.eval()


def predict():
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    generated_img = None
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        generated_img = util.tensor2im(visuals['fake'])
        # image_name = '%s_%s.png' % (name, label)
        # save_path = os.path.join(image_dir, image_name)
    util.save_image(generated_img, images_dir + "results/fake.png")


async def save_upload_file(upload_file):
    with open(os.path.join(images_dir + "testA/real.png"), "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    # Print the submitted file name
    print(f"Submitted file name: {upload_file.filename}")


@app.post("/image")
async def process_image(file: UploadFile = File(...)):
    await save_upload_file(file)
    # image_url = os.path.join(images_dir, file.filename)
    start_time = time()
    predict()
    print("--- %s seconds ---" % (time() - start_time))
    # Return the image URL as a JSON response
    # return JSONResponse(content={"image_name": file.filename})
    return JSONResponse(content={"image_name": "fake.png"})


@app.get("/get_image")
async def read_root():
    return FileResponse(images_dir + "results/fake.png")


# Define a route to handle GET requests to the root URL
@app.get("/")
async def read_root():
    return {"message": "Backend is running"}


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to match your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
