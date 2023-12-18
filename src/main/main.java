package main;

import javax.swing.tree.TreeNode;
import javax.xml.stream.events.Characters;
import java.util.*;

public class main {

    public static void main(String[] args) {
        String s = "51-3";

        //int output = calculateNotWorking(s);
        //int output = calculate1(s);

        //int[][] input={ {0,2}};
        //int[][] input={ {2,1,1}, {1,1,0}, {0,1,2}};
        //int[][] input={ {0,2,2}};
        //int output = orangesRotting(input);
        //List<List<Integer>> immutableList = List.of(List.of(1,3), List.of(3,0,1), List.of(2), List.of(0));
        //boolean output = canVisitAllRooms(immutableList);

        //int[][] input={ {0,0,0}, {1,1,0},{0,0,0}, {0,1,1},{0,0,0} };
        //int output = shortestPath(input, 1);

        //int[] nums1 = {1,2};
        //int[] nums2 = {3,4};
        //double output = findMedianSortedArrays(nums1, nums2);

        //int[] input = {1,-1,0};
        //int output = subarraySum(input, 0);

        //List<List<Integer>> output = combinationSum3(3, 9);

        //int output = minDistance("horse", "ros");
        //int[] input = new int[]{1,2,2,1,1,3};
        //boolean output = uniqueOccurrences(input);

        int[] input = new int[]{1,2,3};
        int output = pivotIndex(input);

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

    public static boolean canVisitAllRooms(List<List<Integer>> rooms) {
        // There are no rooms to search
        if(rooms.isEmpty()) {
            return false;
        }

        Stack<List<Integer>> nodes = new Stack<List<Integer>>();
        HashSet<Integer> visited = new HashSet<>();

        // First room is unlocked
        List<Integer> emptyRoom = rooms.get(0);
        visited.add(0);

        nodes.push(emptyRoom);
        while (!nodes.isEmpty()){
            int size = nodes.size();
            while(size-- > 0) {
                List<Integer> currentKey = nodes.pop();

                // For every room we have the key to
                for(Integer i : currentKey) {
                    // If we have visited the room already
                    if (visited.contains(i)) {
                        continue;
                    }

                    List<Integer> currentRoomItems = rooms.get(i);

                    for(Integer currentRoomItem: currentRoomItems) {
                        // we can now visit the room
                        nodes.add(Arrays.asList(currentRoomItem));
                    }

                    visited.add(i);
                }
            }
        }

        // If we have visited all the rooms possible, return true, else false
        if (visited.size() == rooms.size()) {
            return true;
        }
        return false;
    }

    public static int shortestPath(int[][] grid, int k) {
        int rows = grid.length;
        int columns = grid[0].length;
        int[][] gridCopy = grid;

        // If we only have one row and column, we are at the end
        if (rows == 1 && columns == 1) {
            return 0;
        }

        // We are going to perform BFS search so we use a queue
        Queue<int[]> nodes = new LinkedList<>();
        int[][] visited = new int[rows][columns];
        for (int[] i: visited) {
            Arrays.fill(i, Integer.MAX_VALUE);
        }
        visited[0][0] = 0;

        // Start at top left corner & mark 0,0 as visited
        nodes.offer(new int[]{0,0,0});

        // As simple 2D array to keep track of the directions to take.
        // We can use 4 separate operation, but it is more efficient to use a for-loop to go
        int[][] directions = new int[][] {{0,1},{0,-1},{1,0},{-1,0}};
        int steps = 0;
        while (!nodes.isEmpty()) {
            int n = nodes.size();

            for (int i = 0; i < n; i++) {
                // Get currently where we are
                int[] current = nodes.poll();

                for (int[] direction : directions) {
                    int newX = current[0] + direction[0];
                    int newY = current[1] + direction[1];

                    // Check if we are out of bounds
                    if (newX >= rows || newX < 0 || newY >= columns || newY < 0) {
                        continue;
                    }

                    if (newX == rows - 1 && newY == columns - 1) {
                        return steps + 1;
                    }

                    int newK = current[2] + gridCopy[newX][newY];
                    if (newK > k) {
                        continue;
                    }

                    //  continue if we have more optimal result
                    if (visited[newX][newY] <= newK) {
                        continue;
                    }

                    visited[newX][newY] = newK;
                    nodes.offer(new int[]{newX, newY, newK});
                }
            }
            steps++;
        }

        return -1;
    }

    public static List<Integer> rightSideViewNotWorking(TreeNode root) {
        TreeNode starting = root;
        List<Integer> returnList = new ArrayList<Integer>();
        Stack<TreeNode> nodes = new Stack<>();

        if (root == null) {
            return returnList;
        }

        nodes.push(starting);
        while (!nodes.isEmpty()) {
            TreeNode current = nodes.pop();
            returnList.add(current.val);

            if (current.right != null) {
                nodes.push(current.right);
            }
        }

        return returnList;
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<Integer>();

        helper(root, list, 0);
        return list;
    }

    public void helper(TreeNode node, List<Integer> list, int depth) {
        if (node != null) {
            if (depth == list.size()) {
                list.add(node.val);
            }
            helper(node.right, list, depth + 1);
            helper(node.left, list, depth + 1);
        }
    }

    public int maxLevelSum(TreeNode root) {
        TreeNode starting = root;
        Queue<TreeNode> nodes = new LinkedList<>();

        if (root == null) {
            return 0;
        }

        nodes.offer(starting);
        int maxLevel = 0;
        int maxVal = Integer.MIN_VALUE;
        int currentLevel = maxLevel;

        while (!nodes.isEmpty()) {
            // Keep track of level
            int n = nodes.size();
            int currentLevelVal = 0;
            currentLevel++;

            for(int i = 0; i < n; i++) {
                TreeNode current = nodes.poll();
                currentLevelVal = currentLevelVal + current.val;

                if (current.right != null) {
                    nodes.offer(current.right);
                }

                if (current.left != null) {
                    nodes.offer(current.left);
                }
            }

            // we have finished this depth
            if (currentLevelVal > maxVal) {
                maxVal = currentLevelVal;
                maxLevel = currentLevel;
            }
        }

        return maxLevel;
    }

    public int goodNodesIterativePainInTheAss(TreeNode root) {
//        TreeNode currentRoot = root;
//        int goodNodes = 1;
//
//        Stack<AbstractMap.SimpleEntry<TreeNode, Integer>> nodes = new Stack<>();
//        // Push the current item + max in the tree
//        nodes.push();
//
//        while(!nodes.isEmpty()) {
//            AbstractMap.SimpleEntry<TreeNode, Integer> current = nodes.pop();
//
//        }
//
//        return goodNodes;

        return 0;
    }

    int goodNodes = 0;
    public int goodNodes(TreeNode root) {
        depthFirstSearch(root, Integer.MIN_VALUE);
        return goodNodes;
    }

    public void depthFirstSearch(TreeNode current, int maxValue) {
        int currentMax = maxValue;
        // if the current one is greater than or equal to max value
        if (current.val >= maxValue) {
            goodNodes++;
            currentMax = current.val;
        }

        // Go down the left and right subtree
        if (current.left != null) {
            depthFirstSearch(current.left, currentMax);
        }
        if (current.right != null) {
            depthFirstSearch(current.right, currentMax);

        }
    }

    int totalFound = 0;
    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return 0;
        }

        depthFirstSearchSum(root, targetSum, 0);
        pathSum(root.left, targetSum);
        pathSum(root.right, targetSum);
        return totalFound;
    }

    public void depthFirstSearchSum(TreeNode currentNode, int targetSum, long counter) {
        if (currentNode == null) {
            return;
        }

        long currentCounter = currentNode.val + counter;
        if(currentCounter == targetSum) {
            totalFound++;
        }

        if (currentNode.left != null) {
            depthFirstSearchSum(currentNode.left, targetSum, currentCounter);
        }
        if (currentNode.right != null) {
            depthFirstSearchSum(currentNode.right, targetSum, currentCounter);
        }

    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return CreateBST(nums, 0, nums.length - 1);
    }

    private TreeNode CreateBST(int[] nums, int l, int r) {
        if (l > r) {
            return null;
        }
        int mid = l + (r - l) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = CreateBST(nums, l, mid - 1);
        root.right = CreateBST(nums, mid + 1, r);
        return root;
    }

    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int nums1Length = nums1.length;
        int nums2Length = nums2.length;
        int[] newArray = new int[nums1Length + nums2Length];

        int pointer1 = 0;
        int pointer2 = 0;
        for(int i = 0; i < newArray.length; i++) {
            int m1, m2;
            if (pointer1 >= nums1Length) {
                m1 = Integer.MAX_VALUE;
            } else {
                m1 = nums1[pointer1];
            }

            if (pointer2 >= nums2Length) {
                m2 = Integer.MAX_VALUE;
            } else {
                m2 = nums2[pointer2];
            }

            if (m1 < m2) {
                newArray[i] = nums1[pointer1];
                pointer1++;
            } else {
                newArray[i] = nums2[pointer2];
                pointer2++;
            }
        }

        int median = (nums1Length + nums2Length) / 2;
        // Check if the sum of n and m is odd.
        if ((nums1Length + nums2Length) % 2 == 1){
            return newArray[median];
        } else {
            return (double)(newArray[median-1] + newArray[median]) / 2;
        }

    }

    public static int subarraySum(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

        int totalFound = 0;
        int prefixsum = 0;

        // Initialize the base case
        map.put(0,1);

        for(int i = 0; i < nums.length; i++) {
            prefixsum = prefixsum + nums[i];

            int difference = prefixsum - k;
            if (map.containsKey(difference)) {
                int current = map.get(difference);
                totalFound = totalFound + current;
            }

            if (map.containsKey(prefixsum)) {
                int prefixSumCurrent = map.get(prefixsum);
                map.put(prefixsum, ++prefixSumCurrent);
            } else {
                map.put(prefixsum, 1);
            }
        }

        return totalFound;
    }

    public static List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        List<HashSet<Integer>> resultHashMap = new ArrayList<>();
        backtrack(new ArrayList<>(), result, n ,k, resultHashMap);
        return result;
    }

    public static void backtrack(List<Integer> currentList, List<List<Integer>> result, int n, int k, List<HashSet<Integer>> resultHashMap) {
        if (currentList.size() == k) {
            if (sumsToN(currentList, n)) {
                HashSet<Integer> item =  calcFreq(currentList);
                if (!resultHashMap.contains(item)) {
                    resultHashMap.add(item);
                    result.add(currentList);
                }
            }
            return;
        }

        int[] zeroNine = new int[] {1,2,3,4,5,6,7,8,9};
        for (Integer val : zeroNine) {
            if (!currentList.contains(val) && currentList.size() < k) {
                List<Integer> clone = new ArrayList<>(currentList);
                clone.add(val);
                backtrack(clone, result, n, k, resultHashMap);
            }
        }
     }

    public static boolean sumsToN(List<Integer> list, int n) {
        int current = 0;
        for(Integer i : list) {
            current = current + i;
        }
        if (current == n) {
            return true;
        }
        return false;
    }

    public static HashSet<Integer> calcFreq(List<Integer> result) {
        HashSet<Integer> item = new HashSet<Integer>();
        for (Integer r : result) {
            item.add(r);
        }

        return item;
    }

    public static int minDistance(String word1, String word2) {
        int word1Length = word1.length();
        int word2Length = word2.length();
        char[] word1CharArray = word1.toCharArray();
        char[] word2CharArray = word2.toCharArray();

        int[][] dp = new int[word1Length + 1][word2Length +1];

        // Base case
        for(int i = 0; i <= word2Length; i++) {
            dp[0][i] = i;
        }
        for(int j = 0; j <= word1Length; j++) {
            dp[j][0] = j;
        }

        for(int i = 0; i < word1Length; i++) {
            for(int j = 0; j < word2Length; j++) {
                if (word1CharArray[i] == word2CharArray[j]) {
                    dp[i+1][j+1] = dp[i][j];
                } else {
                    // Insert
                    int insertCost = dp[i][j+1];

                    // Delete
                    int updateCost = dp[i+1][j];

                    // Replace
                    int replaceCost = dp[i][j];

                    int min1 = Math.min(insertCost, updateCost);
                    int min2 = Math.min(replaceCost, min1);
                    dp[i+1][j+1] = min2 + 1;
                }
            }
        }

        return dp[word1Length][word2Length];
    }


    private int answer = Integer.MAX_VALUE;
    private Integer previous = null;

    public int minDiffInBST(TreeNode root) {
        inOrderTraversal(root);
        return answer;
    }

    public void inOrderTraversal(TreeNode root) {
        if (root == null) {
            return;
        }

        inOrderTraversal(root.left);
        if (previous != null) {
            answer = Math.min(answer, root.val - previous);
        }

        previous = root.val;
        inOrderTraversal(root.right);
    }

    int maxIslandSize = 0;
    public int maxAreaOfIsland(int[][] grid) {
        // First we need to find an island
        int x = grid.length;
        int y = grid[0].length;

        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                if (grid[i][j] == 1) {
                    int newMax = bfs(i, j, grid);
                    maxIslandSize = Math.max(newMax, maxIslandSize);
                }
            }
        }

        return maxIslandSize;
    }

    public int bfs(int x, int y, int[][] grid) {
        int innerMaxIsland = 0;
        int gridXRow = grid.length;
        int gridYRow = grid[0].length;
        Queue<int[]> nodes = new LinkedList<>();
        nodes.offer(new int[]{x, y});

        innerMaxIsland++;
        grid[x][y] = 0;

        int[][] directions = new int[][] {{0,1},{0,-1},{1,0},{-1,0}};
        while(!nodes.isEmpty()) {
            int n = nodes.size();
            for (int i = 0; i < n; i++) {
                int[] current = nodes.poll();

                for(int[] direction : directions) {
                    int newX = current[0] + direction[0];
                    int newY = current[1] + direction[1];

                    // We are out of bounds
                    if (newX >= gridXRow || newX < 0 || newY >= gridYRow || newY < 0) {
                        continue;
                    }

                    if (grid[newX][newY] == 1){
                        innerMaxIsland++;
                        grid[newX][newY] = 0;
                        nodes.offer(new int[]{newX, newY});

                    }
                }
            }
        }

        return innerMaxIsland;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode current = root;

        if (current == p || current == q || current == null) {
            return current;
        }

        TreeNode left = lowestCommonAncestor(current.left, p, q);
        TreeNode right = lowestCommonAncestor(current.right, p, q);

        if (left != null && right != null) {
            return current;
        }
        else if (left != null) {
            return left;
        }
        else if (right!=null) {
            return right;
        }
        return null;
    }

    public int diameter;
    public int diameterOfBinaryTree(TreeNode root) {
        diameter = 0;
        dfs(root);
        return diameter;
    }

    public int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        TreeNode current = root;


        int left = dfs(current.left);
        int right = dfs(current.right);

        diameter = Math.max(diameter, left + right);

        return Math.max(left, right) + 1;
    }


    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int max = 0;
        while (left < right) {
            int w = right - left;
            int h = Math.min(height[left], height[right]);
            int area = w * h;
            max = Math.max(max, area);

            if (height[left] < height[right]) {
                left++;
            } else if (height[right] < height[left]) {
                right--;
            } else {
                left++;
                right--;
            }

        }

        return max;
    }

    public static boolean uniqueOccurrences(int[] arr) {
        HashMap<Integer, Integer> dictionary = new HashMap();
        HashSet<Integer> secondary = new HashSet<>();
        for(int item : arr) {
            int count = dictionary.getOrDefault(item, 0);
            count++;
            dictionary.put(item, count);
        }
        for(int value : dictionary.values()){
            if (secondary.contains(value)){
                return false;
            } else {
                secondary.add(value);
            }
        }

        return true;
    }

    public static int largestAltitude(int[] gain) {
        int current = 0;
        int max = 0;
        for(Integer change: gain) {
            current = current + change;
            max = Math.max(max, current);
        }

        return max;
    }

    public static int pivotIndex(int[] nums) {
        int sum = 0;
        int leftSum = 0;
        for (int x: nums) {
            sum = sum + x;
        }
        for (int i = 0; i < nums.length; i++) {
            if (leftSum == sum - leftSum - nums[i])  {
                return i;
            }
            leftSum = leftSum + nums[i];
        }
        return -1;
    }

    public ListNode deleteMiddle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }

        ListNode slow = head;
        ListNode fast = head.next.next;

        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        slow.next = slow.next.next;
        return head;
    }

    public ListNode oddEvenList(ListNode head) {
        if (head == null) {
            return head;
        }

        if (head.next ==  null) {
            return head;
        }

        ListNode even = head.next;

        int counter = 2;
        ListNode first = head;
        ListNode second = head.next;
        while(second.next != null) {
            first.next = second.next;
            first = second;
            second = second.next;
            counter++;
        }

        if (counter % 2 != 0) {
            first.next = null;
            second.next = even;
        } else {
            first.next = even;
            second.next = null;
        }

        return head;
    }

    public int pairSum(ListNode head) {
        ListNode endNode = head;
        ListNode startNode = head;
        int max = 0;
        Stack<ListNode> stack = new Stack<>();

        while(endNode != null) {
            stack.push(endNode);
            endNode = endNode.next;
        }

        while (startNode != null) {
            ListNode end = stack.pop();
            max = Math.max(max, end.val + startNode.val);
            startNode = startNode.next;
        }

        return max;
    }

    class RecentCounter {

        Queue<Integer> queue;
        public RecentCounter() {
            queue = new LinkedList<>();
        }

        public int ping(int t) {
            queue.add(t);
            int lastItem = queue.peek();

            while(lastItem < t - 3000) {
                queue.poll();
                lastItem = queue.peek();
            }

            return queue.size();
        }
    }

    // ** provided classes ** //
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
}
