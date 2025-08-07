/**
 * LLMVM SDK - Node.js Example
 *
 * This example demonstrates how to use the LLMVM SDK in a Node.js environment.
 * Make sure to build the SDK first: npm run build
 */

const { LLMVMClient, user, assistant, system, text } = require('../dist/index.js');

async function main() {
    // Initialize the client
    const client = new LLMVMClient({
        baseUrl: 'http://localhost:8011',
        timeout: 30000
    });

    try {
        // Check connection
        console.log('Checking connection to LLMVM server...');
        const health = await client.health();
        console.log('Server status:', health.status);
        console.log();

        // Example 1: Simple completion
        console.log('Example 1: Simple completion');
        console.log('----------------------------');

        const response1 = await client.complete([
            system('You are a helpful assistant.'),
            user('What is the capital of France?')
        ], {
            model: 'gpt-5',
            temperature: 1.0,
            maxTokens: 8192
        });

        console.log('Response:', response1.getText());
        console.log();

        // Example 2: Multi-turn conversation
        console.log('Example 2: Multi-turn conversation');
        console.log('----------------------------------');

        const thread = await client.createThread([
            system('You are a knowledgeable history teacher.'),
            user('Tell me about the French Revolution.'),
        ]);

        console.log('Created thread:', thread.id);

        // First response
        const result1 = await client.toolsCompletions(thread);
        console.log('Assistant:', result1.messages[result1.messages.length - 1].content[0].sequence);

        // Add follow-up question
        await client.addMessages(thread.id, [
            user('What were the main causes?')
        ]);

        // Get second response
        const result2 = await client.toolsCompletions(
            await client.getThread(thread.id)
        );
        console.log('\nFollow-up response:', result2.messages[result2.messages.length - 1].content[0].sequence);
        console.log();

        // Example 3: Streaming response
        console.log('Example 3: Streaming response');
        console.log('-----------------------------');

        let streamedContent = '';
        await client.complete([
            user('Write a haiku about programming.')
        ], {
            model: 'gpt-5',
            temperature: 1.0,
            onChunk: (chunk) => {
                // In a real application, you might update UI here
                if (chunk.token) {
                    streamedContent += chunk.token;
                    process.stdout.write(chunk.token);
                }
            }
        });
        console.log('\n');

        // Example 4: Python execution
        console.log('Example 4: Python execution');
        console.log('---------------------------');

        const pythonThread = await client.createThread();
        const pythonResult = await client.executePython(pythonThread.id, `
import math

# Calculate some values
radius = 5
area = math.pi * radius ** 2
circumference = 2 * math.pi * radius

result = {
    'radius': radius,
    'area': area,
    'circumference': circumference
}
result
        `);

        console.log('Python execution result:');
        console.log('Variable:', pythonResult.var_name);
        console.log('Value:', pythonResult.var_value);
        console.log();

        // Example 5: Working with different content types
        console.log('Example 5: Content types');
        console.log('------------------------');

        // Create different content types
        const textContent = text('This is plain text content.');
        const userWithMultipleContent = user([
            text('I have a question about this text:'),
            text('What does it mean to be a good programmer?')
        ]);

        console.log('Text content:', textContent.getText());
        console.log('User message:', userWithMultipleContent.getText());
        console.log();

        // Example 6: Error handling
        console.log('Example 6: Error handling');
        console.log('-------------------------');

        try {
            // Try to get a non-existent thread
            await client.getThread(999999);
        } catch (error) {
            console.log('Expected error:', error.message);
        }

        // Cleanup: Get all threads
        console.log('\nListing all threads:');
        const allThreads = await client.getThreads();
        allThreads.forEach(t => {
            console.log(`- Thread ${t.id}: ${t.messages.length} messages`);
        });

    } catch (error) {
        console.error('Error:', error);
    }
}

// Run the examples
main().catch(console.error);