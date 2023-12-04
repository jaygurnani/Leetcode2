package main;

import javax.xml.stream.events.Characters;
import java.util.*;

public class main {

    public static void main(String[] args) {
        String s = "51-3";

        //int output = calculateNotWorking(s);
        //int output = calculate1(s);

        //int[][] input={ {0,2}};
        int[][] input={ {2,1,1}, {1,1,0}, {0,1,2}};
        //int[][] input={ {0,2,2}};
        int output = orangesRotting(input);
        System.out.println(output);
    }

    public static int calculateNotWorking(String s) {
        char[] sCharArray = s.toCharArray();

        Stack<Integer> intStack = new Stack<>();
        Stack<Character> operands = new Stack<>();

        for (int i = 0; i < sCharArray.length; i++) {
            char sCurrentChar = sCharArray[i];

            if (operands.size() > 0) {
                int right = Integer.parseInt(String.valueOf(sCurrentChar));
                int left = intStack.pop();
                char operand = operands.pop();
                if (operand == '+') {
                    intStack.push(right + left);
                }
            }

            else if (sCurrentChar == '(') {
                StringBuilder sb = new StringBuilder();
                for (int j = i + 1; j < sCharArray.length; j++) {
                    char jCurrentChar = sCharArray[j];
                    if (jCurrentChar == ')'){
                        j++;
                        int subProblemValue = calculateNotWorking(sb.toString());
                        intStack.push(subProblemValue);
                    }
                    sb.append(jCurrentChar);
                }
            }

            else if (sCurrentChar == '+') {
                operands.push(sCurrentChar);
            } else {
                intStack.push(Integer.parseInt(String.valueOf(sCurrentChar)));
            }


        }

        return intStack.pop();
    }

    public static int calculate1(String s) {
        Stack<Integer> stack = new Stack<Integer>();
        int result = 0;
        int number = 0;
        int sign = 1;

        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);

            if (Character.isDigit(c)) {
                // We multiple by 10 here because the digit might consist of 2 values e.g.
                // 51 and we will read this 10 * 0 then 10 * 5 which will give us the 51
                number = 10 * number + (int)(c - '0');
            } else if (c == '+') {
                result = result + sign * number;
                number = 0;
                sign = 1;
            } else if (c == '-') {
                result = result + sign * number;
                number = 0;
                sign = -1;
            } else if (c == '(') {
                // Push the current state and reset everything back to 0
                stack.push(result);
                stack.push(sign);

                // Reset the values
                sign = 1;
                result = 0;
            } else if (c == ')') {
                result = result + sign * number;
                number = 0;

                // sign before the parenthesis
                result = result * stack.pop();
                // result calculated before the parenthesis
                result = result + stack.pop();
            }
        }

        if(number != 0) {
            result = result + sign * number;
        }
        return result;
    }

    // breath first search
    public static int nearestExit(char[][] maze, int[] entrance) {
        int rows = maze.length;
        int columns = maze[0].length;

        Queue<int[]> nodes = new LinkedList<>();

        // We don't need a visited since we can just use the maze itself
        Queue<int[]> visited = new LinkedList<>();

        nodes.offer(entrance);
        maze[entrance[0]][entrance[1]] = '+';

        // As simple 2D array to keep track of the directions to take.
        // We can use 4 separate operation, but it is more efficient to use a for-loop to go
        int[][] directions = new int[][] {{0,1},{0,-1},{1,0},{-1,0}};

        int x, y;
        int steps = 0;
        while (!nodes.isEmpty()) {
            steps++;
            int n = nodes.size();

            for(int i = 0; i < n; i++) {
                int[] current = nodes.poll();

                for(int j = 0; j < directions.length; j++) {
                    x = current[0] + directions[j][0];
                    y = current[1] + directions[j][1];

                    // check if we are out of bounds
                    if (x < 0 || x >= rows || y < 0 || y >= columns) {
                        continue;
                    }

                    // We are at the wall
                    if (maze[x][y] == '+'){
                        continue;
                    }

                    // If this direction is empty, not visited and is at the boundary, we have arrived at the exit.
                    if (x == 0 || x == rows - 1 || y == 0 || y == columns - 1) {
                        return steps;
                    }

                    // Otherwise, we change this direction as visited and put into the queue to check at the next step.
                    maze[x][y] = '+';
                    nodes.offer(new int[] {x, y} );
                }


            }
        }

        // If all the possible nodes and directions checked but no exits found, return -1.
        return -1;
    }

    public static int orangesRotting(int[][] grid) {
        // First we have to find the rotten oranges
        int rows = grid.length;
        int columns = grid[0].length;
        int[][] gridCopy = grid;

        // Proceed to do BFS
        Queue<int[]> nodes = new LinkedList<>();
        int freshOrangeCount = 0;

        for(int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (grid[i][j] == 2) {
                    nodes.offer(new int[]{i,j});
                } else if (grid[i][j] == 1){
                    freshOrangeCount++;
                }
            }
        }

        if (freshOrangeCount == 0){
            return 0;
        }

        if (nodes.isEmpty()){
            return -1;
        }


        // Current rotten X and Y and level
        int intCurrentLevel = -1;
        int[][] directions = new int[][] {{0,1},{0,-1},{1,0},{-1,0}};

        while(!nodes.isEmpty()) {
            int size = nodes.size();
            while (size-- > 0) {
                int[] current = nodes.poll();
                int currentX = current[0];
                int currentY = current[1];

                for (int[] direction : directions) {
                    int newX = currentX + direction[0];
                    int newY = currentY + direction[1];

                    if (newX < 0 || newX >= rows || newY < 0 || newY >= columns) {
                        continue;
                    }

                    // See if the current orange is already rotten, do nothing
                    if (gridCopy[newX][newY] == 2) {
                        continue;
                    }
                    // if the current grid is empty, do nothing
                    if (gridCopy[newX][newY] == 0){
                        continue;
                    }

                    // if the current grid is not rotten, mark it as rotten
                    if (gridCopy[newX][newY] == 1){
                        gridCopy[newX][newY] = 2;

                        freshOrangeCount--;
                        // add this to a new place to start looking from
                        nodes.offer(new int[]{newX, newY});

                    }
                }

            }
            intCurrentLevel++;
        }


        if (freshOrangeCount == 0)
            return intCurrentLevel;
        return -1;
    }
}
