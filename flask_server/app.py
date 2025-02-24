from os import system
system("clear")
from flask import Flask, request, jsonify, render_template,make_response
from flask_restful import Api, Resource
from modules import evaluate

app = Flask(__name__)
api = Api(app)


# Variables to store the cumulative data
total_cost:float = 0.0
total_queries:int = 0

# API Resource to handle JSON data submission
class DataSubmission(Resource):
    def post(self):
        global total_cost, total_queries

        data = request.get_json()

        # Extract cost and queries from the request
        cost = data.get('cost', 0.0)
        queries = data.get('queries', 0)

        # Update the total cost and queries
        total_cost += cost
        total_queries += queries

        # Print the data for verification (optional)
        print("Received Data:")
        # print(f"Questions: {data.get('question')}")
        # print(f"Contexts: {data.get('contexts')}")
        # print(f"Answer: {data.get('answer')}")
        # print(f"Ground Truth: {data.get('ground_truth')}")
        print(f"Cost: {cost}")
        print(f"Queries: {queries}")

        test_set = {"question":data.get('question'),"contexts":data.get('contexts'),"answer":data.get('answer'),"ground_truth":data.get('ground_truth')}

        result,df,image_path = evaluate.evaluate_rag(test_set)
        print("-----------------------------------")
        print(df)
        print(result)
        image_path = request.host_url+image_path
        print(f"Image Path: {image_path}")
        print("-----------------------------------")
        return jsonify({"message": result,"image_path":image_path})

# Route to display the dashboard in HTML
class Home(Resource):
    def get(self):
        global total_cost, total_queries
        
        # Render the HTML template with total cost and queries
        return make_response(render_template('index.html', total_cost=total_cost, total_queries=total_queries))

# Add the resource to the API
api.add_resource(DataSubmission, '/submit_data')
api.add_resource(Home, '/')


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",threaded=True,port=5000)
