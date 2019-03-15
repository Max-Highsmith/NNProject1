import pytest
import run_experiments as rx
import warnings

@pytest.fixture
def input_dim():
	return 28


#Experiments parameters
@pytest.fixture
def exp3_params():
	param_dict =  {
		"epochs"              : "10",
		"order"		      : "shuffle",
		"activation_function" : 'tanh',
		"learning_rate"       : 0.0001,
		"initialization"      : 'uniform',
		"data_presentation"   : "seiral",
		"loss_fn"             : "SquaredError",
		"project_part"        : 2,
		"trail_num"           : 2,
		"freezeP" 	      : False,
		"numFilters" 	      : 16,
		"filterDim"           : 7
	}
	return param_dict


@pytest.fixture
def exp2_params():
	param_dict = {
		"epochs"  		: 10,
		"data_presentation"	:'serial',
		"order"			:'shuffle',
		"activation_function"	:"tanh",
		"initialization"	:"uniform",
		"learning_rate"		:"0.001",
		"project_part"		: 2,
		"loss_fn"		:"SquaredError",
		"freezeP"		: False,
		"filterDim"		: 28,
		"numFilters"		: 2,
		"trail_num"		: 0
		}
	return param_dict



#TODO
class Test_Class_Activation_Functions():
	def test_sigmoid(self):
		assert True

	def test_tanhActivation(self):
		assert True

	def test_rel(self):
		assert True

#TODO
class Test_Class_Loss_Functions():
	def test_Squared_Error(self):
		assert True
	def test_Cross_Entropy(self):
		assert True


class Test_Class_Setup_Functions():
	@pytest.mark.parametrize('experiment_params, expected_receptive_field',[
				(exp2_params(), 1),
				(exp3_params(), 22)])
	def test_getReceptiveField(self, experiment_params, input_dim, expected_receptive_field):
		filterDim = experiment_params["filterDim"]
		assert expected_receptive_field == rx.getReceptiveField(input_dim, filterDim)


	def test_setUpNetwork(self):
		assert True
		
class Test_Class_Network_Layers():

	def test_perceptron_init(self):
		#TODO
		assert True

	def test_perceptron_forwardPass(self):
		#TODO
		assert True

	def test_perceptrion_backProp(self):
		#TODO
		assert True
	
	def test_conv_filter_init(self):
		pass
		#fit = experiment_params["filterDim"]
		#test = prop_init
		#assert test!=fit
		#assert True

	def test_conv_filter_forwardPass(self):
		#TODO
		assert True

	def test_conv_filter_backProp(self):
		#TODO
		assert True

	def test_combo_init(self):
		#TODO
		assert True

	def test_combo_forwardPass(self):
		#TODO
		assert True

	def test_combo_backwardPasS(self):
		#TODO
		assert True	

	def test_decisionLayer_forwardPass(self):
		assert True
		
