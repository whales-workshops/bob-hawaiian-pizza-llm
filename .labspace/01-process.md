# Fine tuning of Qwen2.5-0.5B-Instruct to make an Hawaiian Pizza expert


1. Build the container image using the following command:

    ```bash
    docker build -t fine-tuning .
    ```

    It will likely take a few moments to download the base image and perform all the required updates.

2. Start a container using the newly created image and mounting the `gguf_output` directory (where the script will put the final GGUF file):

    ```bash
    docker run -ti \
        --gpus all \
        -v ./output:/usr/local/app/gguf_output \
        -v ./data:/usr/local/app/data \
        fine-tuning
    ```

    You will see the fine-tuning process kick off, which may have moments where it appears the output has frozen. The full model training takes about 12 minutes to run.

5. Once the fine-tuning finishes, you should see the GGUF file in the `output` directory:

    ```bash
    ls output/
    ```

    You should see a file named `qwen2.5-0.5b-instruct.F16.gguf`

Hooray! You now have a model!


Now that you have completed the fine-tuning process, you need to validate the fine-tuning. Then, you'll be ready to publish the model.

## 🧪 Validate the model

Validating a model can be a lengthy process, as you want to ensure the model works for a variety of inputs it wasn't trained with.

For the purposes of this lab, you will simply run a new input against the model to validate it works as expected.

1. Load the GGUF model into Docker Model Runner by using the following `docker model package` command:

    ```bash
    docker model package --gguf $PWD/output/bob-hawaiian-pizza.F16.gguf demo/bob-hawaiian-pizza:0.5b-0.0.1
    ```

    This will give the name `demo/bob-hawaiian-pizza:0.5b-0.0.1` to the model

2. Validate the model is available with the Docker Model Runner by listing the model:

    ```bash
    docker model list
    ```

3. Send a new input against the model by using the `docker model run` command:

    ```bash
    docker model run demo/bob-hawaiian-pizza:0.5b-0.0.1 "What is the best pizza in the world?"
    ```

Hooray! It looks like your model is working as expected!


## 📦 Publish the model

There are a variety of ways you can publish your model. By using the `docker model package` command, you will package the model as an OCI (Open Container Initiative) Artifact. This provides a few benefits:

- Distribute the model using the same registries you use for your container images
- The same credentials that pull container images will pull your models, simplifying deployment setups
- This artifact can be easily used by the Docker Model Runner

1. To publish your model, run the following command after swapping `DOCKER_USERNAME` with your username:

    ```bash no-run-button
    #log in to docker hub first
    docker login -u k33g
    docker model package --gguf $PWD/output/bob-hawaiian-pizza.F16.ggu k33g/bob-hawaiian-pizza:0.5b-0.0.1 --push

    docker model run k33g/bob-hawaiian-pizza:0.5b-0.0.1 "What is the best pizza in the world?"
    ```

    You should notice this command is the same as what you ran before but with a different name and the addition of the `--push` flag.

    


