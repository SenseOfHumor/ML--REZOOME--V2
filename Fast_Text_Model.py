import fasttext

# Train the model
model = fasttext.train_supervised(input="fast_train.txt", epoch=25, lr=0.1)

# Save the model
model.save_model("skills_model.bin")
