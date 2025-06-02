export interface ParsedSegment {
  type: 'text' | 'helpers_block' | 'result_block';
  content: string;
  isComplete: boolean;
}

export function parseHelperTags(input: string): ParsedSegment[] {
  const segments: ParsedSegment[] = [];
  
  // Regular expressions for complete blocks
  const helpersBlockPattern = /<helpers>([\s\S]*?)<\/helpers>/g;
  const resultBlockPattern = /<helpers_result>([\s\S]*?)<\/helpers_result>/g;
  
  // Track all matches
  interface BlockMatch {
    type: 'helpers' | 'result';
    start: number;
    end: number;
    fullMatch: string;
    content: string;
  }
  
  const matches: BlockMatch[] = [];
  
  // Find all complete blocks
  let match;
  while ((match = helpersBlockPattern.exec(input)) !== null) {
    matches.push({
      type: 'helpers',
      start: match.index,
      end: match.index + match[0].length,
      fullMatch: match[0],
      content: match[1]
    });
  }
  
  while ((match = resultBlockPattern.exec(input)) !== null) {
    matches.push({
      type: 'result',
      start: match.index,
      end: match.index + match[0].length,
      fullMatch: match[0],
      content: match[1]
    });
  }
  
  // Sort matches by position
  matches.sort((a, b) => a.start - b.start);
  
  // Process the input
  let currentIndex = 0;
  
  for (const blockMatch of matches) {
    // Add any text before this block
    if (blockMatch.start > currentIndex) {
      const textContent = input.substring(currentIndex, blockMatch.start);
      if (textContent) {
        segments.push({
          type: 'text',
          content: textContent,
          isComplete: true
        });
      }
    }
    
    // Add the block with tags included
    segments.push({
      type: blockMatch.type === 'helpers' ? 'helpers_block' : 'result_block',
      content: blockMatch.fullMatch,
      isComplete: true
    });
    
    currentIndex = blockMatch.end;
  }
  
  // Handle remaining content and incomplete blocks
  if (currentIndex < input.length) {
    const remaining = input.substring(currentIndex);
    
    // Check for incomplete helpers block
    if (remaining.includes('<helpers>') && !remaining.includes('</helpers>')) {
      const helperStart = remaining.indexOf('<helpers>');
      
      // Add text before incomplete block
      if (helperStart > 0) {
        segments.push({
          type: 'text',
          content: remaining.substring(0, helperStart),
          isComplete: true
        });
      }
      
      // Add incomplete helpers block
      segments.push({
        type: 'helpers_block',
        content: remaining.substring(helperStart),
        isComplete: false
      });
    }
    // Check for incomplete result block
    else if (remaining.includes('<helpers_result>') && !remaining.includes('</helpers_result>')) {
      const resultStart = remaining.indexOf('<helpers_result>');
      
      // Add text before incomplete block
      if (resultStart > 0) {
        segments.push({
          type: 'text',
          content: remaining.substring(0, resultStart),
          isComplete: true
        });
      }
      
      // Add incomplete result block
      segments.push({
        type: 'result_block',
        content: remaining.substring(resultStart),
        isComplete: false
      });
    }
    // Check for incomplete opening tags
    else if (/<help(?:e(?:r(?:s)?)?)?$/.test(remaining) || 
             /<helpers_(?:r(?:e(?:s(?:u(?:l(?:t)?)?)?)?)?)?$/.test(remaining)) {
      const tagMatch = remaining.match(/<help(?:e(?:r(?:s)?)?)?$|<helpers_(?:r(?:e(?:s(?:u(?:l(?:t)?)?)?)?)?)?$/);
      if (tagMatch && tagMatch.index !== undefined) {
        // Add text before incomplete tag
        if (tagMatch.index > 0) {
          segments.push({
            type: 'text',
            content: remaining.substring(0, tagMatch.index),
            isComplete: true
          });
        }
        
        // Add incomplete tag as text
        segments.push({
          type: 'text',
          content: remaining.substring(tagMatch.index),
          isComplete: false
        });
      }
    }
    // Otherwise it's just text
    else if (remaining) {
      segments.push({
        type: 'text',
        content: remaining,
        isComplete: true
      });
    }
  }
  
  return segments;
}

// Helper function to reconstruct the original text from segments
export function reconstructFromSegments(segments: ParsedSegment[]): string {
  return segments.map(segment => segment.content).join('');
}

// Helper function to check if content has any helper tags
export function hasHelperTags(input: string): boolean {
  return input.includes('<helpers>') || 
         input.includes('<helpers_result>') ||
         input.includes('</helpers>') ||
         input.includes('</helpers_result>') ||
         /<help(?:e(?:r(?:s)?)?)?$/.test(input) ||
         /<helpers_(?:r(?:e(?:s(?:u(?:l(?:t)?)?)?)?)?)?$/.test(input);
}