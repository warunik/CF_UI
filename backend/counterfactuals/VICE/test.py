import data
import model
import vice

path = "diabetes.csv"
model_path = 'finalized_model.sav'

d = data.Data(path)
m = model.Model().load_model(model_path)

v = vice.Vice(d,m)

v.generate_explanation(10)