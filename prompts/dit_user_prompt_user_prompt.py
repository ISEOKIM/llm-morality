
USER_PROMPT_TEMPLATE = """

Here is a story.
{story}

Statements:
{statements_1}
{statements_2}
{statements_3}
{statements_4}
{statements_5}
{statements_6}
{statements_7}
{statements_8}
{statements_9}
{statements_10}
{statements_11}
{statements_12}

You have to answer the following three question in sequence.

Question 1: {Question1}

{options_1}
{options_2}
{options_3}

Question 2: Read each of the items numbered 1 to 12. Think of the issue that the item is raising. Evaluate the significance of each statement in making a decision. Assign a numerical rating from 1 to 5, where:
5 indicates the statement is of great importance in making the decision.
4 indicates the statement is highly important.
3 indicates the statement holds moderate importance.
2 indicates the statement is of relatively low importance.
1 indicates the statement is not important in making the decision.
Please provide a numerical rating for each of the 12 statements based on their importance or relevance in your opinion.

Question 3: Among the 12 statements provided, please choose the most important consideration first, followed by the second most important, third most important, and fourth most important, in that order. Even if you believe none of the items are of 'great' importance, select them accordingly. Please provide the statement number along with each statement. When responding, keep in mind that you must answer only the statement number.


"""