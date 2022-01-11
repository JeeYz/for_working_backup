package models.RecordVoice;

public class ConvertExample {
    static float exampleNum = -1.05f;

    public static void main(String[] args) {
        System.out.println(exampleNum);
        System.out.println(Float.floatToIntBits(exampleNum));

        int temp = Float.floatToIntBits(exampleNum);
        System.out.println(temp);
        System.out.println(temp);

        byte[] btemp = intToByteArray(temp);
        System.out.println(btemp);
        System.out.println(btemp.length);

        int itemp = byteArrayToInt(btemp);
        System.out.println(itemp);
        
        float ftemp = Float.intBitsToFloat(itemp);
        System.out.println(ftemp);

        System.out.println(Float.SIZE);
        System.out.println(Integer.SIZE);
        System.out.println(Byte.SIZE);
    }

    public static  byte[] intToByteArray(int value) {
		byte[] byteArray = new byte[4];
		byteArray[0] = (byte)(value >> 24);
		byteArray[1] = (byte)(value >> 16);
		byteArray[2] = (byte)(value >> 8);
		byteArray[3] = (byte)(value);
		return byteArray;
	}

    public static int byteArrayToInt(byte bytes[]) {
        return ((((int)bytes[0] & 0xff) << 24) |
                (((int)bytes[1] & 0xff) << 16) |
                (((int)bytes[2] & 0xff) << 8) |
                (((int)bytes[3] & 0xff)));
    } 

}